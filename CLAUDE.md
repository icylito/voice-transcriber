# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local transcription pipeline, entirely on-device via WhisperX (faster-whisper + wav2vec2 alignment) on
an NVIDIA GPU — no cloud API, no server, no third-party diarization model. Four ways to feed it audio,
all converging on the same transcribe → align → render pipeline:

- Drop a video into `videos/` and run `transcribe.py` (or `--pick` one interactively) — batch mode.
- `--record`: capture mic + system (loopback) audio to a buffer, transcribe on stop.
- `--url URL`: `yt-dlp` downloads the audio, then it flows through the same path as a dropped video.
- `--live`: the same mic+system capture, streamed in rolling chunks with incremental console output.

This CLAUDE.md documents what exists today. `future_plan.md` holds the (not-scheduled) direction for
a future Windows GUI on top of this same pipeline — check it for context if asked about a GUI.

Every mode writes a timestamped `.md` to `transcripts/`; a `.srt` is written alongside it only if
`output.write_srt = true` in `config.toml` (default `false`). `--record` and `--live` additionally
label each block `(user)` (mic) or `(system)` (loopback) since there are two live audio sources;
file/URL-based modes have only one source and stay unlabeled.

Two optional local-LLM extras, both via Ollama, both toggleable in `config.toml`'s `[llm]` table and
both fail-safe (never break a transcription run if Ollama isn't available): punctuation/capitalization
cleanup (`llm.enabled`), and content-based transcript titling (`llm.auto_title`). See `_restore_punctuation`
and `_generate_title` in Architecture below.

## Commands

```bash
# `voice` is voice.bat: it resolves ffmpeg onto PATH, then runs transcribe.py, then returns
# straight to the prompt (no "press any key" pause -- terminal scrollback already keeps the
# output, so there's nothing to hold the window open for).
# If you add the repo root to your User PATH, `voice` (no .bat, no .\ prefix) works from any
# directory, in any shell -- an ollama-style CLI feel. If it's not resolving (PATH not set up,
# or a fresh clone), fall back to `.\voice` from inside the repo, or the full `voice.bat`; both
# always work regardless of PATH.
voice                          # process every new file in videos/
voice path\to\one.mp4          # transcribe a single file (unquoted spaces in the path are fine)
voice --pick                   # interactively choose one pending video from videos/
voice --record                 # record mic+system audio, transcribe on stop
voice --url <URL>              # download audio via yt-dlp, then transcribe it
voice --live                   # live chunked transcription, streamed to console
voice --list                   # show pending/done videos, don't transcribe
voice --list-devices           # show mic/loopback device indices for config.toml
voice --model quality          # force the quality model regardless of mode
voice --model fast             # force the fast model regardless of mode
voice --help                   # list every command and flag

# Equivalent without the wrapper (needs ffmpeg already on PATH)
.venv\Scripts\python.exe transcribe.py --record

# Install/reinstall deps into the venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Note on PATH changes**: adding a directory to User PATH only affects *new* processes started after
the change — an already-open terminal (or anything spawned from one) keeps its environment as of when
it launched. After changing PATH, open a genuinely new terminal window to see the update; a new tab or
subprocess of an existing session won't pick it up.

`--record`, `--url`, `--live`, `--pick`, `--list`, `--list-devices`, and a positional `file` argument are
all mutually exclusive. The `file` positional uses `nargs="*"` and gets rejoined with spaces
(`Path(" ".join(args.file))`) rather than the usual `nargs="?"` single-token positional — PowerShell users
routinely paste an unquoted file path containing spaces (game-clip folder names, timestamps in filenames,
etc.), which the shell splits into multiple argv tokens before argparse ever sees it; rejoining recovers
the original path instead of erroring with "unrecognized arguments". A properly quoted path still works
identically (argparse hands it over as a single-element list either way).

There is no build step, linter, or test suite in this repo — it's a handful of scripts plus a config file.

`ffmpeg` must be on `PATH` (installed system-wide via `winget install Gyan.FFmpeg`). A shell opened before
that install won't see it; a fresh terminal will. `ffmpeg` is also what `yt-dlp` shells out to for the
`-x --audio-format wav` extraction step in `fetch.py`.

## Architecture

Four files, split by concern rather than one script, since capture, live streaming, and URL fetching are
genuinely distinct pieces of logic:

- **`transcribe.py`** — entry point, config loading, the transcribe→align→render pipeline, and CLI wiring.
- **`capture.py`** — mic + WASAPI loopback (system audio) capture, shared by `--record` and `--live`.
- **`fetch.py`** — `yt-dlp` wrapper for `--url`.
- **`live.py`** — the rolling-chunk transcription loop for `--live`.

### Config and two models

**`load_config()`** reads `config.toml` at import time into the module-level `CONFIG` dict, and every
tunable is read from it into module-level constants. **Don't hardcode values that belong in
`config.toml`** — that file exists specifically so behavior can be tuned without touching code.

There are **two Whisper models**, both loaded via the same WhisperX/faster-whisper code path (this is a
config-level swap, not a second inference stack):

- `model.quality.name` (`large-v3`, full model) — used for batch/`--pick`/single-file/`--record`/`--url`.
  No real-time constraint on those paths, so accuracy wins. `large-v3` needs ~5.5GB VRAM at `float16` —
  check yours with `nvidia-smi` before assuming it fits; on a smaller card, drop to a smaller Whisper
  variant or switch `model.compute_type` to `int8`/`int8_float16`.
- `model.fast.name` (`large-v3-turbo`) — used only for `--live`, where per-chunk latency is what makes it
  feel real-time.
- `main()` picks between them via `_model_name_for(mode, override)`; `--model quality|fast` forces one
  regardless of mode.

### The pipeline (`run_pipeline` / `run_dual_pipeline`)

- **`_transcribe_and_align(audio, asr_model, align_cache)`** is the shared core: `asr_model.transcribe()`
  (faster-whisper/CTranslate2, with VAD-based silence filtering built into WhisperX's pipeline already —
  this is what keeps hallucination-on-silence artifacts down, no extra flag needed) produces rough
  segments, then `whisperx.align()` runs a wav2vec2 forced-alignment pass for word-level timing. The
  alignment model is cached per-language in `align_cache` across a run. Alignment processes segments
  **one at a time on the GPU with no progress output of its own** (a `# TODO: ... batched inference`
  comment sits right above that loop in whisperx's own source) — for audio with thousands of short
  segments (dialogue-heavy games/anime, etc.) this can legitimately run for a long time in total
  silence, which looks exactly like a hang. `_run_with_heartbeat()` wraps the `whisperx.align()` call
  and prints a `...still aligning (Ns elapsed)` line every 20s so that's distinguishable from an
  actual freeze; `_transcribe_and_align` also prints the segment count up front so the scale of what's
  coming is visible before alignment starts.
  **Confirmed by direct testing this isn't the only cause of apparent freezes**: a 2-minute gameplay
  DVR clip that had appeared to hang for hours produced only 5 segments and re-ran in seconds once
  retested in isolation — nowhere near enough segments for the per-segment alignment loop above to
  explain it. Since DVR clips come from actively-played games, GPU contention from the still-running
  game (sharing the same GPU as `large-v3`) is the more likely explanation for that specific case.
  `_warn_if_gpu_busy()` — called right before each
  `whisperx.load_model()` in `main()` — checks `torch.cuda.mem_get_info()` (whole-device memory, so
  it picks up usage from *other* processes, not just this one) and prints a warning if over half the
  GPU's memory is already in use, rather than failing silently slow. Best-effort: any exception is
  swallowed so a diagnostic check never blocks an actual run.
- **`run_pipeline(name, audio, asr_model, align_cache)`** — single audio source (a decoded video/URL
  file). Used by `transcribe_file()`.
- **`run_dual_pipeline(name, mic_audio, system_audio, asr_model, align_cache)`** — the `--record` path
  only (`--live` has its own loop in `live.py` and never calls this). Transcribes mic and system audio
  **independently** (two separate `_transcribe_and_align` calls, not one mixed-down stream), tags the
  resulting segments `speaker="user"` / `speaker="system"`, and merges both lists by `start` time before
  rendering. Either side is skipped if empty (e.g. nothing played through speakers during a `--record`
  session) — feeding a zero-length array into WhisperX's VAD step raises a confusing `pyannote`
  `ValueError` about tensor shape, so both call sites guard on `audio.size` first.
- **`_restore_punctuation(label, segments)`** — optional cleanup pass, gated on `llm.enabled` in
  `config.toml` (default `true`). Sends segment text to a local LLM over Ollama's `/api/chat` (stdlib
  `urllib`, no new dependency) to fix punctuation/capitalization/question marks after Whisper's own pass,
  mutating segment texts in place. Called from both `run_pipeline` and `run_dual_pipeline`, right after
  `_transcribe_and_align` — **not** called from `--live`'s path (`live.py`'s `run()` writes straight to
  `_write_transcript`), since a synchronous LLM round-trip per rolling window would defeat live mode's
  whole reason for skipping alignment.
  - Segments are sent in numbered batches of `llm.batch_size` (default 40) rather than one HTTP call per
    segment, both for speed on dialogue-heavy audio (hundreds/thousands of short segments) and to stay
    within the model's context window. Model choice: `qwen3:4b-instruct-2507-q4_K_M` — confirmed by
    direct testing that Ollama's plain `qwen3:4b` tag is a different, hybrid-reasoning model that burns
    ~30-40s "thinking" per call even with `think: false` and Qwen's own `/no_think` soft-switch both set
    (neither suppressed it); the `-instruct-2507` tag is a non-thinking release of the same model family
    and answers in ~2s with equivalent quality — pull that one specifically, not `qwen3:4b`.
  - **Fails safe, never fails the run**: any batch where the request errors, times out, or comes back
    with the wrong number of lines is skipped entirely (original text kept for every segment in that
    batch). On top of that, confirmed by direct testing that even with an explicit "don't add/remove
    words" instruction, this model can still slip in a small connector word (e.g. "living in Seattle" came
    back once as "and living in Seattle") — so each corrected line's word count (via `_word_count()`,
    a punctuation-stripped token count) is compared against the original before being accepted; a mismatch
    falls back to Whisper's original text for that one line rather than trusting a fabricated word. A
    transcript is a record of what was actually said, so a wrong word is worse than a missing comma.
- **`_ensure_llm_available()`** — runs at most once per invocation, called from `main()` right after
  the `--list` early return and before any mode branch, but only when `llm.enabled or llm.auto_title`
  (so `--list`/`--list-devices`, or a config with both off, never touch Ollama at all). Health-checks
  `GET {llm.host}/` (Ollama's documented liveness endpoint, distinct from the `/api/chat` calls the
  two LLM features make below). If down and `llm.auto_start` is on, prompts `Ollama is not running.
  Do you want to start it? [y/N]:` (plain `input()`, same convention as `pick_video()`/
  `prompt_filename()`). `y` launches `ollama serve` detached (`subprocess.Popen` with
  `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP` so it outlives this process — verified by direct
  testing that it survives after the launching process exits) and polls the health endpoint (wrapped
  in `_run_with_heartbeat`) up to `llm.startup_timeout_seconds`. Any failure — declined, timeout,
  `ollama` missing from PATH — sets the module-level `_llm_runtime_available` flag to `False` for
  **this run only**; `config.toml`'s toggles are never touched. `_llm_usable_for(feature_enabled)`
  (`feature_enabled and _llm_runtime_available`) is what `_restore_punctuation` and `_generate_title`
  both check, so the "config toggle AND this run's Ollama state" combination lives in one place.
- **`_generate_title(segments)`** — LLM-based title generation, gated by `llm.auto_title` (default
  `true`, independent of `llm.enabled`'s punctuation gate) and `_llm_usable_for()`. One `/api/chat`
  call (unlike `_restore_punctuation`'s numbered batches — a title needs the whole transcript's gist,
  not a sliding window) over the first ~4000 characters of joined segment text, asking for a 5-8 word
  Title Case description of what the content is — meeting, casual chat, gameplay commentary, etc. —
  inferred by the model from the content itself, no separate classifier. Output is sanitized
  (`_strip_invalid_filename_chars`, shared with `prompt_filename`) and slugified
  (`_slugify_title`) to match `_suggest_filename`'s `snake_case` convention. Returns `None` on any
  failure (unusable, too-short transcript, request error/timeout, empty response) — this is a new
  first link prepended onto `_write_transcript`'s existing fallback chain, not a replacement for it.
- **`_write_transcript(name, segments, *, skip_llm_title=False)`** — shared tail. `skip_llm_title=True`
  (set only by the `--url` flow, since `fetch.download_audio()`'s returned path's `.stem` is already
  yt-dlp's real pulled title) uses `name` directly with no suggestion logic at all. Otherwise:
  `_generate_title(segments) or _suggest_filename(segments, ...) or name`. `prompt_filename()`
  (interactive `input()`, defaulting to that suggestion if left blank) lets the user accept or
  override it either way. Then `render_markdown()`, always written; `render_srt()` is written too
  only when `WRITE_SRT` (`output.write_srt` in `config.toml`, default `false`) is set.
  `winsound.MessageBeep()` (stdlib, no new dependency) signals completion audibly so unattended jobs
  don't need to be watched.
- **`_suggest_filename(segments, max_words)`** — offline, dependency-free content-based filename
  guesser: counts non-stopword words across the whole transcript (`_STOPWORDS` is a small built-in
  English filler-word list — "the", "um", "gonna", etc.), keeps the `max_words` most frequent
  (`output.suggested_filename_words` in `config.toml`, default 5), then re-orders just those top
  words by where they first appear in the transcript so the result reads like a short phrase (e.g.
  `wuthering_waves_boss_fight_build`) rather than a frequency-sorted word bag. Returns `None` on an
  empty/too-short transcript so the caller falls back to its own default.
- **`transcribe_file(path, ..., skip_llm_title=False)`** — thin wrapper: `whisperx.load_audio()` shells
  out to the system `ffmpeg` binary to decode (not PyAV — `faster-whisper`'s PyAV path is never reached
  since we always hand it a pre-decoded array), then `run_pipeline(..., skip_llm_title=skip_llm_title)`,
  then marks `.done`. `main()`'s `--url` branch is the only caller that passes `skip_llm_title=True`.

### Grouping and rendering

- **`group_by_pause()`** merges consecutive segments into paragraph-sized blocks. It breaks to a new
  block on *any* of: a silence gap ≥ `pause_gap_seconds`, the current block running past
  `max_block_seconds` (conversational audio may never produce a real pause), or **a speaker change** —
  the last condition exists so a user/system exchange doesn't get silently merged into one mislabeled
  block. Segments without a `speaker` key (file/URL mode) compare as `None == None`, so single-source
  grouping is unaffected.
- **`render_markdown()`** formats each group as `**[start - end] (speaker)** text` — the `(speaker)` tag
  is omitted entirely when `speaker` is falsy.
- **`render_srt()`** — same grouping, standard numbered-cue SRT format, `(speaker) ` prefixed into the
  cue text when present.

### Single-file path resolution

**`_resolve_file_path(path)`** guards against a common Windows typo: Explorer hides known file
extensions by default, so a path typed or pasted by hand can be missing its real extension (e.g.
`...09.DVR` when the actual file is `...09.DVR.mp4`). If the given path doesn't exist as-is, it
tries appending each extension in `output.video_extensions` before giving up — the raw ffmpeg
"No such file or directory" error this replaces doesn't explain *why* the file wasn't found.

### Batch scanning and `--pick`

**`find_pending_videos()`** scans `videos/` and skips any file with a marker in `.rendered/` (named
`{video filename}.done`, empty file, created after a successful write). Re-running `transcribe.py` is
idempotent — it only processes new drops. `--list` prints pending vs. done without transcribing anything.
`output.video_extensions` in `config.toml` covers both video *and* audio containers (`.mp4`/`.mov`/etc.
alongside `.mp3`/`.wav`/`.m4a`/etc.) — no code path needed to change for audio support, since
`whisperx.load_audio()` already shells out to `ffmpeg`, which decodes both uniformly; the only thing
gating audio files out before was the extension allowlist.
With `--pick`, `main()` calls `find_pending_videos()` then `pick_video()`, which prints a numbered list
and reads a choice via `input()`. Plain `input()` was chosen over an arrow-key menu library
(`questionary`/`prompt_toolkit`) because that stack requires a real attached Win32 console screen buffer
and hard-errors under Git Bash/mintty and other non-conhost terminals.

### `capture.py`: mic + system audio, kept separate, on two different backends

The mic and loopback streams are **never mixed down** — they stay separate end to end specifically so
`run_dual_pipeline` can label who said what. `list_devices()` backs `--list-devices`;
`record.mic_device` / `record.loopback_device` in `config.toml` pick specific devices (`-1` = system
default).

- **Loopback (system audio)** goes through PyAudioWPatch/WASAPI — the only practical way to record
  system audio on Windows (exclusive-mode streams can't loopback, so this opens a shared-mode input
  stream on the loopback analogue of the default output device). This backend works reliably.
- **Mic** goes through an **`ffmpeg` subprocess** (`-f dshow -i audio="<device name>"`), not
  PyAudioWPatch, as of the fix below. `_resolve_mic()` still resolves the device via the WASAPI host
  API (consistent with `--list-devices`' enumeration) but only to read off the device's friendly
  name — the actual stream is opened by ffmpeg, not PortAudio.

**Why two backends**: on the original dev machine, PortAudio (which both PyAudioWPatch *and* a
separately-tested vanilla `sounddevice` install are built on) failed to open one specific USB mic
(a HyperX QuadCast 2) under every host API it exposes (WASAPI, MME, WDM-KS alike) with generic
`-9999`/`-9996` ("Unanticipated host error" / "Invalid device") errors — confirmed via direct testing
this was a PortAudio-specific incompatibility with that particular device, not a code bug, mute/
permission/host-API-mismatch issue, or OS-level block: the same device opened fine through Windows'
own recorder app and through `ffmpeg -f dshow`. Rather than special-case around it, mic capture goes
through `ffmpeg` **unconditionally, for every user** — `ffmpeg` is already a hard dependency of this
project, it's simpler than juggling two capture paths, and it sidesteps this whole class of
PortAudio/device incompatibility regardless of which mic you have. If your mic *does* work fine with
PyAudioWPatch, you won't notice any difference; this just means you'll never hit the bug above either.
`_start_ffmpeg_mic_capture()` asks ffmpeg to output already-mono, already-resampled-to-
`record.sample_rate` raw `s16le` PCM directly (`-ac 1 -ar <rate> -f s16le -`), so `_StreamBuffer`'s own
resample step is a no-op for the mic path; a background thread pumps ffmpeg's stdout into the same
`_StreamBuffer.write()` the loopback callback uses. Stopping tries a graceful `q` over stdin first (what
ffmpeg treats as "quit"), falling back to `kill()` if that doesn't land in time.

`_StreamBuffer` accumulates each stream's raw samples, downmixes to mono, and resamples (via
`scipy.signal.resample_poly`, already a transitive dependency — no new package needed) to
`record.sample_rate` (a no-op for the mic path per above, since ffmpeg already delivers that rate). It
optionally trims old audio once it exceeds `max_native_samples`: record mode leaves this unbounded
(reads once, on stop), live mode bounds it to roughly `chunk_seconds + overlap_seconds` so repeated
per-step reads stay cheap regardless of session length — an unbounded buffer re-concatenated every
`step_seconds` would grow quadratically over a long live session.

### `live.py`: rolling-chunk transcription

Live mode reuses **`faster-whisper` via WhisperX in Python**, not whisper.cpp's `stream` example — the
latter needs its own GGML model file (a different format than the CTranslate2 model already in use) and
its own CMake+Visual Studio+CUDA Toolkit native build, a second toolchain on a machine that's already hit
real native-DLL pain (see "Known environment gotchas" below). Reusing WhisperX means the exact model,
engine, and GPU pipeline already working, at the cost of hand-writing the chunk/overlap reconciliation.

- Every `step_seconds`, pull the last `chunk_seconds` of **each** stream from the live capture buffer and
  run `asr_model.transcribe()` on each independently — no `whisperx.align()` here; forced alignment's
  latency would defeat a low-latency mode, and live captions don't need word-perfect timestamps.
- Each window is a fresh, independent `transcribe()` call with no state carried forward, so there's no
  cross-window conditioning/hallucination-drift risk to guard against — the risk that pattern usually
  causes only applies when previous text is explicitly fed back into a later call, which this design
  never does.
- **Overlap reconciliation** (`_new_text` / `_StreamReconciler`): since consecutive windows share audio,
  diff the new window's transcript against the previous window's transcript with
  `difflib.SequenceMatcher.find_longest_match()` and only emit the non-overlapping remainder. Verified
  against real transcribed speech (not just synthetic strings) that this doesn't duplicate text across
  steps.
- Prints `[user] ...` / `[system] ...` incrementally to the console; on stop (Enter, mirroring the
  project's other `input()`-based prompts), returns the accumulated speaker-tagged segments so
  `main()` can run them through the same `_write_transcript()` as every other mode.
- **`_ConsoleWriter`** governs *when* a new console line starts. Printing a fresh line on every
  `step_seconds` tick (1s by default) reads as way too fast/jittery, scrolling mid-sentence. Instead
  it reuses the same `pause_gap_seconds` / `max_block_seconds` thresholds the saved transcript's
  paragraph grouping (`group_by_pause()` in `transcribe.py`) already uses: same speaker + no real
  pause + block not yet too long → the new piece just extends the current line (`print(..., end="")`);
  otherwise it closes the line and starts a new `[speaker] ...` one. `run()` takes
  `pause_gap_seconds`/`max_block_seconds` as parameters (passed in from `transcribe.py`'s `CONFIG`)
  rather than importing `transcribe` directly, to avoid a circular import.

### `fetch.py`: yt-dlp

`download_audio(url, dest_dir)` shells out to `yt-dlp -x --audio-format wav ... --print
after_move:filepath` (via `[sys.executable, "-m", "yt_dlp", ...]`, not a bare `yt-dlp` call — the venv's
`Scripts/` dir isn't necessarily on `PATH` when launched through `voice.bat`) and parses the final
resolved path from stdout. Downloads to a temp file rather than piping: `yt-dlp` can't combine `-x`
(audio extraction) with `-o -` (stdout piping), and piping raw video into another process is fragile
(throttling, seeking, format-detection issues) — this mirrors the temp-file pattern whisper.cpp's own
`yt-wsp.sh` script uses. The downloaded file is handed straight to `transcribe_file()`, unchanged.

**The printed path isn't always trustworthy on Windows**: confirmed by direct testing that `--print
after_move:filepath` can report a filename that differs from what actually lands on disk — e.g. a
video title containing a literal `?` (illegal in Windows filenames) gets swapped for a lookalike
fullwidth `？` in the real on-disk filename, but the un-swapped name is still what gets printed.
`download_audio()` snapshots `dest_dir` before running yt-dlp; if the printed path doesn't exist
after the run, it diffs the directory for whatever new file with a matching extension actually
appeared, rather than trying to replicate yt-dlp's own Windows filename sanitization rules.

## Known environment gotchas

Windows + NVIDIA-GPU specific gotchas encountered during development. The first two apply to anyone
installing this on Windows; the rest were hit on the original dev machine and may or may not apply to
yours — kept here in case you hit the same symptoms.

- `torch` needs to be pinned to `2.8.0+cu128` (installed explicitly from `download.pytorch.org/whl/cu128`
  *before* `pip install whisperx`, since `whisperx` requires `torch~=2.8.0` and the default PyPI `torch`
  wheel on Windows is CPU-only — installing whisperx first would silently lose GPU acceleration).
- The `torchcodec is not installed correctly` warning on every run is expected and harmless: pyannote's
  VAD step is fed a pre-decoded waveform dict (`whisperx/asr.py`), so it never actually needs torchcodec's
  file-decoding path.
- **If Windows Smart App Control blocks native DLLs** pulled in by `pip install`ed ML packages (PyAV,
  scipy have no Microsoft reputation yet), you'll see `ImportError: DLL load failed ... Application
  Control policy` on `whisperx`/`transformers` imports. Disabling Smart App Control (there's no per-app
  exception mechanism) is the workaround used during development.
- **If PyAudioWPatch/PortAudio fails to open your mic** under every host API (WASAPI/MME/WDM-KS) with
  generic `-9999`/`-9996` errors even though the mic works fine elsewhere (Windows' own recorder,
  `ffmpeg -f dshow`) — this happened on the dev machine with one specific USB mic (HyperX QuadCast 2),
  confirmed as a PortAudio-specific device incompatibility, not a code/permission/host-API issue. This
  shouldn't actually affect you either way: mic capture already goes through `ffmpeg` unconditionally
  for every user (see "Why two backends" above), specifically so this class of bug is sidestepped
  regardless of which mic you have.
