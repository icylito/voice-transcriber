"""Transcribe videos dropped in videos/, or capture audio directly, into timestamped Markdown
(+ SRT) transcripts in transcripts/.

All tunable settings live in config.toml (models, device, paths, etc.) - no code changes needed
to tweak them.

Usage:
    voice                       # process every new file in videos/
    voice path\\to\\one.mp4      # transcribe a single file (spaces in the path don't need quoting)
    voice --pick                # interactively choose one pending video
    voice --record              # record mic+system audio, transcribe on stop
    voice --url URL             # download audio via yt-dlp, then transcribe it
    voice --live                # live chunked transcription with console output
    voice --list                # show pending/done videos without transcribing
    voice --list-devices        # show capture device indices for config.toml
    voice --model quality       # force the quality (large-v3) model regardless of mode
    voice --model fast          # force the fast (large-v3-turbo) model regardless of mode

Flags --record, --url, --live, --pick, --list, --list-devices, and a positional file argument
are all mutually exclusive with each other.
"""

import argparse
import json
import logging
import re
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.request
import warnings
import winsound
from pathlib import Path

# These are all known-harmless per CLAUDE.md's "Known environment gotchas": the torchcodec
# UserWarning (VAD is fed a pre-decoded waveform, never needs torchcodec's file-decoding path),
# the TF32 reproducibility notice, and INFO-level logging from whisperx/pyannote/pytorch_lightning
# (language detection, VAD start, checkpoint-version upgrades). None of them indicate a problem,
# so silence them here rather than showing 30-40 lines of expected noise on every run.
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import torch
import whisperx

import capture
import fetch
import live

ROOT = Path(__file__).parent
STATE_DIR = ROOT / ".rendered"  # marks which videos have already been processed
INVALID_FILENAME_CHARS = '<>:"/\\|?*'


def load_config() -> dict:
    with open(ROOT / "config.toml", "rb") as f:
        return tomllib.load(f)


CONFIG = load_config()
VIDEOS_DIR = ROOT / CONFIG["paths"]["videos_dir"]
TRANSCRIPTS_DIR = ROOT / CONFIG["paths"]["transcripts_dir"]
VIDEO_EXTENSIONS = set(CONFIG["output"]["video_extensions"])

DEVICE = CONFIG["model"]["device"]
COMPUTE_TYPE = CONFIG["model"]["compute_type"]
BATCH_SIZE = CONFIG["model"]["batch_size"]
LANGUAGE = CONFIG["model"]["language"] or None
MODEL_QUALITY_NAME = CONFIG["model"]["quality"]["name"]
MODEL_FAST_NAME = CONFIG["model"]["fast"]["name"]

PAUSE_GAP_SECONDS = CONFIG["output"]["pause_gap_seconds"]
MAX_BLOCK_SECONDS = CONFIG["output"]["max_block_seconds"]
WRITE_SRT = CONFIG["output"].get("write_srt", False)
SUGGESTED_FILENAME_WORDS = CONFIG["output"].get("suggested_filename_words", 5)

RECORD_SAMPLE_RATE = CONFIG["record"]["sample_rate"]
RECORD_MIC_DEVICE = CONFIG["record"]["mic_device"]
RECORD_LOOPBACK_DEVICE = CONFIG["record"]["loopback_device"]

LIVE_CHUNK_SECONDS = CONFIG["live"]["chunk_seconds"]
LIVE_STEP_SECONDS = CONFIG["live"]["step_seconds"]
LIVE_OVERLAP_SECONDS = CONFIG["live"]["overlap_seconds"]

LLM_ENABLED = CONFIG["llm"]["enabled"]
LLM_HOST = CONFIG["llm"]["host"]
LLM_MODEL = CONFIG["llm"]["model"]
LLM_BATCH_SIZE = CONFIG["llm"]["batch_size"]
LLM_TIMEOUT_SECONDS = CONFIG["llm"]["timeout_seconds"]


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_srt_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    h, remainder = divmod(total_ms, 3_600_000)
    m, remainder = divmod(remainder, 60_000)
    s, ms = divmod(remainder, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def group_by_pause(segments: list[dict], gap_seconds: float, max_block_seconds: float) -> list[dict]:
    """Merge consecutive same-speaker segments into one block until a silence gap is crossed, the
    block has been running long enough that it's worth breaking anyway (conversational audio with
    quick back-and-forth replies may never produce a real pause), or the speaker changes (mic vs.
    system audio, when segments are tagged)."""
    groups: list[dict] = []
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        speaker = seg.get("speaker")
        if groups:
            pause = seg["start"] - groups[-1]["end"]
            block_duration = seg["end"] - groups[-1]["start"]
            same_speaker = groups[-1]["speaker"] == speaker
        if groups and pause < gap_seconds and block_duration <= max_block_seconds and same_speaker:
            groups[-1]["text"] += " " + text
            groups[-1]["end"] = seg["end"]
        else:
            groups.append({"start": seg["start"], "end": seg["end"], "text": text, "speaker": speaker})
    return groups


def render_markdown(title: str, segments: list[dict]) -> str:
    lines = [f"# {title}", ""]
    for group in group_by_pause(segments, PAUSE_GAP_SECONDS, MAX_BLOCK_SECONDS):
        time_range = f"{format_timestamp(group['start'])} - {format_timestamp(group['end'])}"
        speaker_tag = f" ({group['speaker']})" if group["speaker"] else ""
        lines.append(f"**[{time_range}]{speaker_tag}** {group['text']}")
        lines.append("")
    return "\n".join(lines)


def render_srt(segments: list[dict]) -> str:
    lines = []
    for i, group in enumerate(group_by_pause(segments, PAUSE_GAP_SECONDS, MAX_BLOCK_SECONDS), start=1):
        speaker_tag = f"({group['speaker']}) " if group["speaker"] else ""
        lines.append(str(i))
        lines.append(f"{format_srt_timestamp(group['start'])} --> {format_srt_timestamp(group['end'])}")
        lines.append(f"{speaker_tag}{group['text']}")
        lines.append("")
    return "\n".join(lines)


def _resolve_file_path(path: Path) -> Path:
    """Windows Explorer hides known file extensions by default, so a path typed or pasted by
    hand (e.g. '...DVR' instead of the real '...DVR.mp4') is a common typo. If the given path
    doesn't exist as-is, try appending each known extension before giving up -- the raw ffmpeg
    'No such file or directory' error this would otherwise produce doesn't explain why."""
    if path.exists():
        return path
    for ext in sorted(VIDEO_EXTENSIONS):
        candidate = path.with_name(path.name + ext)
        if candidate.exists():
            print(f"Note: '{path.name}' not found -- using '{candidate.name}' instead "
                  f"(Windows often hides file extensions).")
            return candidate
    return path


def marker_path(video_path: Path) -> Path:
    return STATE_DIR / f"{video_path.name}.done"


def find_pending_videos() -> list[Path]:
    pending = []
    for path in sorted(VIDEOS_DIR.iterdir()):
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if not marker_path(path).exists():
            pending.append(path)
    return pending


def print_status() -> None:
    all_videos = sorted(p for p in VIDEOS_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
    if not all_videos:
        print(f"No videos in {VIDEOS_DIR}")
        return
    pending = set(find_pending_videos())
    for path in all_videos:
        status = "pending" if path in pending else "done"
        print(f"  [{status}] {path.name}")


# Common English filler/function words, excluded when guessing a filename from transcript
# content -- otherwise "the", "so", "um" etc. would dominate the word-frequency count instead of
# whatever the recording is actually about.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "because", "as", "of", "at", "by",
    "for", "with", "about", "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "than", "too", "very", "just", "i", "me", "my", "myself", "we", "our", "ours",
    "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs",
    "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "ok", "okay", "yeah", "yep", "yes", "um", "uh", "uhh", "umm", "hmm", "like", "gonna", "wanna",
    "gotta", "really", "actually", "basically", "literally", "kind", "sort", "know", "mean",
    "guess", "right", "well", "thing", "things", "stuff", "get", "got", "going", "go", "one",
    "two", "also", "still", "even", "much", "many",
}


def _suggest_filename(segments: list[dict], max_words: int = 5) -> str | None:
    """Best-effort filename guessed from what was actually said: count non-filler words across
    the whole transcript, keep the most frequent ones, then output them in the order they first
    appear (not frequency order) so the result reads like a short phrase, e.g.
    'wuthering_waves_boss_fight_build', instead of a shuffled word bag. Returns None if the
    transcript is too short/empty to produce anything meaningful, so the caller can fall back to
    its own default name."""
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    position = 0
    for seg in segments:
        for word in re.findall(r"[A-Za-z']+", seg["text"].lower()):
            word = word.strip("'")
            if len(word) < 3 or word in _STOPWORDS:
                continue
            counts[word] = counts.get(word, 0) + 1
            first_seen.setdefault(word, position)
            position += 1
    if not counts:
        return None
    top_words = sorted(counts, key=lambda w: (-counts[w], first_seen[w]))[:max_words]
    top_words.sort(key=lambda w: first_seen[w])
    return "_".join(top_words) or None


def prompt_filename(default: str) -> str:
    entered = input(f"What do you want the file name to be? [{default}]: ").strip()
    if not entered:
        return default
    cleaned = "".join(c for c in entered if c not in INVALID_FILENAME_CHARS).strip()
    return cleaned or default


def pick_video(pending: list[Path]) -> Path | None:
    print("Pending videos:")
    for i, path in enumerate(pending, start=1):
        print(f"  {i}. {path.name}")
    entered = input(f"Pick a video [1-{len(pending)}]: ").strip()
    if not entered.isdigit() or not (1 <= int(entered) <= len(pending)):
        return None
    return pending[int(entered) - 1]


def _load_align_model(language: str, align_cache: dict):
    if language not in align_cache:
        print(f"loading alignment model for language={language}")
        align_cache[language] = whisperx.load_align_model(language_code=language, device=DEVICE)
    return align_cache[language]


def _run_with_heartbeat(label: str, func, interval: float = 20.0):
    """Runs func() while printing a heartbeat every `interval` seconds. Some steps (notably
    whisperx's forced-alignment pass, which processes segments one at a time on the GPU with no
    progress output of its own) can legitimately run for a long time in total silence -- easy to
    mistake for a hang, especially on dialogue-heavy audio (games, anime) with thousands of short
    segments. This doesn't make it faster, just makes it obvious it's still alive."""
    done = threading.Event()
    start = time.monotonic()

    def _beat():
        while not done.wait(interval):
            print(f"  ...still {label} ({int(time.monotonic() - start)}s elapsed)")

    threading.Thread(target=_beat, daemon=True).start()
    try:
        return func()
    finally:
        done.set()


def _transcribe_and_align(audio, asr_model, align_cache: dict) -> list[dict]:
    result = asr_model.transcribe(audio, batch_size=BATCH_SIZE, language=LANGUAGE)
    segment_count = len(result["segments"])
    print(f"  transcribed {segment_count} segments (language={result['language']})")
    align_model, align_metadata = _load_align_model(result["language"], align_cache)
    print(f"  aligning {segment_count} segments -- this happens one at a time, so audio with "
          f"lots of short lines (dialogue-heavy games/anime, etc.) can take a while")
    result = _run_with_heartbeat(
        "aligning",
        lambda: whisperx.align(
            result["segments"], align_model, align_metadata, audio, DEVICE, return_char_alignments=False
        ),
    )
    return result["segments"]


_PUNCTUATION_SYSTEM_PROMPT = (
    "You will be given a numbered list of raw speech-transcript lines, one line of speech per "
    "number. For each line, add correct capitalization, commas, periods, and question marks "
    "where a real person would naturally use them when speaking. Do not add, remove, or change "
    "any words, and do not merge, split, or reorder lines. Output ONLY the same numbered list "
    "back, one corrected line per number, nothing else."
)


def _llm_punctuate_batch(lines: list[str]) -> list[str] | None:
    """Sends one numbered batch of raw segment texts to the local LLM (via Ollama's /api/chat)
    and returns the corrected lines in the same order. Returns None on any failure (Ollama not
    running, timeout, bad response, or a line count that doesn't match what was sent) so the
    caller can fall back to leaving that batch's original Whisper text untouched -- this is a
    nice-to-have polish step, never something the pipeline should hard-fail on."""
    numbered = "\n".join(f"{i}: {line}" for i, line in enumerate(lines, start=1))
    payload = json.dumps(
        {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": _PUNCTUATION_SYSTEM_PROMPT},
                {"role": "user", "content": numbered},
            ],
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{LLM_HOST}/api/chat", data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=LLM_TIMEOUT_SECONDS) as response:
            result = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None
    content = result.get("message", {}).get("content", "").strip()
    if not content:
        return None
    corrected = [
        re.sub(r"^\d+[.:]\s*", "", line.strip()) for line in content.splitlines() if line.strip()
    ]
    if len(corrected) != len(lines):
        return None
    return corrected


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text.lower()))


def _restore_punctuation(label: str, segments: list[dict]) -> None:
    """Best-effort punctuation/capitalization cleanup via a local LLM, opt-in via llm.enabled in
    config.toml (see CLAUDE.md). Mutates segment texts in place. Segments are sent in batches of
    LLM_BATCH_SIZE rather than one at a time -- dialogue-heavy audio can produce hundreds/
    thousands of short segments (see the alignment note above), and one HTTP round-trip per
    segment would make that dramatically slower than it needs to be, plus a giant single request
    risks overflowing the model's context window. Not used by --live mode: that path already
    skips forced alignment specifically to stay low-latency, and a synchronous LLM call per
    rolling window would defeat the same goal.

    Even with an explicit "don't add/remove words" instruction, small LLMs occasionally slip in
    a connector word (confirmed by direct testing: "living in Seattle" once came back as "and
    living in Seattle"). A transcript is supposed to be a faithful record of what was actually
    said, so a wrong word is worse than a missing comma -- each corrected line's word count is
    compared against the original, and any line that doesn't match keeps Whisper's original
    text instead of the LLM's rewrite."""
    if not LLM_ENABLED or not segments:
        return
    texts = [seg["text"].strip() for seg in segments]
    if not any(texts):
        return
    print(f"[{label}] restoring punctuation for {len(texts)} segments via {LLM_MODEL}...")
    skipped = 0
    for start in range(0, len(texts), LLM_BATCH_SIZE):
        batch = texts[start : start + LLM_BATCH_SIZE]
        corrected = _llm_punctuate_batch(batch)
        if corrected is None:
            skipped += len(batch)
            continue
        for offset, text in enumerate(corrected):
            original = batch[offset]
            if _word_count(text) != _word_count(original):
                skipped += 1
                continue
            segments[start + offset]["text"] = text
    if skipped:
        print(
            f"[{label}] Note: kept original punctuation for {skipped} segment(s) -- the LLM "
            f"call failed, returned a mismatched response, or changed the word count (is Ollama "
            f"running? try: ollama list)."
        )


def _write_transcript(name: str, segments: list[dict]) -> None:
    suggested = _suggest_filename(segments, SUGGESTED_FILENAME_WORDS) or name
    filename = prompt_filename(suggested)
    md_path = TRANSCRIPTS_DIR / f"{filename}.md"
    md_path.write_text(render_markdown(filename, segments), encoding="utf-8")
    written = [md_path]
    if WRITE_SRT:
        srt_path = TRANSCRIPTS_DIR / f"{filename}.srt"
        srt_path.write_text(render_srt(segments), encoding="utf-8")
        written.append(srt_path)
    winsound.MessageBeep()
    print(f"[{name}] wrote " + " and ".join(str(p) for p in written))


def run_pipeline(name: str, audio, asr_model, align_cache: dict) -> None:
    print(f"[{name}] transcribing")
    segments = _transcribe_and_align(audio, asr_model, align_cache)
    _restore_punctuation(name, segments)
    _write_transcript(name, segments)


def run_dual_pipeline(name: str, mic_audio, system_audio, asr_model, align_cache: dict) -> None:
    print(f"[{name}] transcribing (user)")
    user_segments = _transcribe_and_align(mic_audio, asr_model, align_cache) if mic_audio.size else []
    for seg in user_segments:
        seg["speaker"] = "user"

    print(f"[{name}] transcribing (system)")
    system_segments = _transcribe_and_align(system_audio, asr_model, align_cache) if system_audio.size else []
    for seg in system_segments:
        seg["speaker"] = "system"

    merged = sorted(user_segments + system_segments, key=lambda s: s["start"])
    _restore_punctuation(name, merged)
    _write_transcript(name, merged)


def transcribe_file(path: Path, asr_model, align_cache: dict) -> None:
    print(f"[{path.name}] loading audio")
    audio = whisperx.load_audio(str(path))
    run_pipeline(path.stem, audio, asr_model, align_cache)
    marker_path(path).touch()


def _warn_if_gpu_busy() -> None:
    """A large chunk of GPU/VRAM already in use by another process (most plausibly a game still
    running, since DVR clips come straight from actively playing one) can make transcription and
    especially alignment run dramatically slower than usual without any error -- just silence.
    torch.cuda.mem_get_info() reports whole-device memory, not just this process's share, so it
    picks up contention from *other* processes too. Best-effort only: never blocks a run."""
    if DEVICE != "cuda":
        return
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        used_gb = (total_bytes - free_bytes) / (1024**3)
        total_gb = total_bytes / (1024**3)
        if used_gb / total_gb > 0.5:
            print(
                f"Note: {used_gb:.1f}/{total_gb:.1f} GB of GPU memory is already in use by other "
                f"processes (e.g. a game still running) -- transcription and alignment can run "
                f"much slower than usual while sharing the GPU under load."
            )
    except Exception:
        pass


def _model_name_for(mode: str, override: str | None) -> str:
    if override == "quality":
        return MODEL_QUALITY_NAME
    if override == "fast":
        return MODEL_FAST_NAME
    return MODEL_FAST_NAME if mode == "live" else MODEL_QUALITY_NAME


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="voice", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "file",
        nargs="*",
        help="Transcribe a single file instead of scanning videos/ "
        "(spaces in the path don't need quoting -- unquoted words are auto-joined)",
    )
    parser.add_argument(
        "--pick", action="store_true", help="Interactively choose one pending video from videos/ to transcribe"
    )
    parser.add_argument("--record", action="store_true", help="Record mic+system audio, transcribe on stop")
    parser.add_argument("--url", metavar="URL", help="Download audio from a URL via yt-dlp, then transcribe it")
    parser.add_argument("--live", action="store_true", help="Live chunked transcription with console output")
    parser.add_argument("--list", action="store_true", help="List pending/done videos in videos/ and exit")
    parser.add_argument("--list-devices", action="store_true", help="List mic/loopback capture devices and exit")
    parser.add_argument("--model", choices=["quality", "fast"], help="Override the mode's default model choice")
    args = parser.parse_args()
    file_path = _resolve_file_path(Path(" ".join(args.file))) if args.file else None

    chosen_modes = [bool(file_path), args.pick, args.record, bool(args.url), args.live, args.list, args.list_devices]
    if sum(chosen_modes) > 1:
        parser.error("--record, --url, --live, --pick, --list, --list-devices, and a file argument are mutually exclusive")

    if args.list_devices:
        capture.list_devices()
        return

    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    if args.list:
        print_status()
        return

    if args.live:
        model_name = _model_name_for("live", args.model)
        _warn_if_gpu_busy()
        print(f"Loading Whisper model ({model_name}) on {DEVICE}...")
        asr_model = whisperx.load_model(model_name, DEVICE, compute_type=COMPUTE_TYPE)
        try:
            segments = live.run(
                asr_model,
                RECORD_SAMPLE_RATE,
                RECORD_MIC_DEVICE,
                RECORD_LOOPBACK_DEVICE,
                LIVE_CHUNK_SECONDS,
                LIVE_STEP_SECONDS,
                LIVE_OVERLAP_SECONDS,
                BATCH_SIZE,
                LANGUAGE,
                PAUSE_GAP_SECONDS,
                MAX_BLOCK_SECONDS,
            )
            if segments:
                _write_transcript("live", segments)
            else:
                print("No speech captured.")
        except Exception as exc:
            print(f"[live] FAILED: {exc}", file=sys.stderr)
        return

    model_name = _model_name_for("batch", args.model)
    _warn_if_gpu_busy()
    print(f"Loading Whisper model ({model_name}) on {DEVICE}...")
    asr_model = whisperx.load_model(model_name, DEVICE, compute_type=COMPUTE_TYPE)
    align_cache: dict = {}

    if args.record:
        try:
            mic_audio, system_audio = capture.record_until_enter(
                RECORD_SAMPLE_RATE, RECORD_MIC_DEVICE, RECORD_LOOPBACK_DEVICE
            )
            run_dual_pipeline("recording", mic_audio, system_audio, asr_model, align_cache)
        except Exception as exc:
            print(f"[recording] FAILED: {exc}", file=sys.stderr)
        return

    if args.url:
        try:
            downloaded = fetch.download_audio(args.url, VIDEOS_DIR)
            transcribe_file(downloaded, asr_model, align_cache)
        except Exception as exc:
            print(f"[{args.url}] FAILED: {exc}", file=sys.stderr)
        return

    if file_path:
        targets = [file_path]
    elif args.pick:
        pending = find_pending_videos()
        if not pending:
            print(f"No new videos to transcribe in {VIDEOS_DIR}")
            return
        chosen = pick_video(pending)
        if chosen is None:
            print("Cancelled.")
            return
        targets = [chosen]
    else:
        targets = find_pending_videos()
        if not targets:
            print(f"No new videos to transcribe in {VIDEOS_DIR}")
            return

    for path in targets:
        try:
            transcribe_file(path, asr_model, align_cache)
        except Exception as exc:
            print(f"[{path.name}] FAILED: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
