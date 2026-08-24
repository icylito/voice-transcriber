"""Live chunked transcription: rolling mic+system capture, transcribed in small overlapping
windows so text streams to the console roughly every step_seconds, since Whisper-family models
don't transcribe live by default. Each window is an independent transcribe() call (no shared
state/conditioning across windows), so windows are reconciled after the fact by diffing each new
window's text against the previous one and only emitting the non-overlapping remainder -- this
also means no cross-window hallucination drift, since nothing is fed forward between calls.

No word-level alignment here (unlike the batch/record/url pipeline) -- alignment's forced-alignment
pass adds latency that defeats the point of a low-latency mode, and live captions don't need
word-perfect timestamps, just readable streamed text plus a roughly-timed saved transcript.
"""

import difflib
import threading
import time

import capture


def _new_text(prev_text: str, new_text: str) -> str:
    """Portion of new_text beyond whatever prev_text already covered, found via the longest
    matching block between the two overlapping windows' transcripts."""
    if not prev_text:
        return new_text
    matcher = difflib.SequenceMatcher(None, prev_text, new_text)
    match = matcher.find_longest_match(0, len(prev_text), 0, len(new_text))
    if match.size == 0:
        return new_text
    return new_text[match.b + match.size :].lstrip()


class _ConsoleWriter:
    """Keeps live console output readable. Without this, a new line gets printed every single
    step_seconds (1s by default) even mid-sentence, which reads as way too fast/jittery. Instead,
    this only starts a new line when the speaker changes or a real pause/long block happens
    (reusing the same pause_gap_seconds / max_block_seconds thresholds the saved transcript's
    paragraph grouping uses) -- otherwise each new piece just extends the current line, like
    normal streaming captions."""

    def __init__(self, pause_gap_seconds: float, max_block_seconds: float):
        self._pause_gap_seconds = pause_gap_seconds
        self._max_block_seconds = max_block_seconds
        self._last_speaker: str | None = None
        self._last_end: float | None = None
        self._block_start: float | None = None

    def emit(self, speaker: str, text: str, elapsed: float) -> None:
        if not text:
            return
        gap = elapsed - self._last_end if self._last_end is not None else None
        block_duration = elapsed - self._block_start if self._block_start is not None else None
        continues_line = (
            speaker == self._last_speaker
            and gap is not None and gap < self._pause_gap_seconds
            and block_duration is not None and block_duration <= self._max_block_seconds
        )
        if continues_line:
            print(f" {text}", end="", flush=True)
        else:
            if self._last_speaker is not None:
                print()  # close out the previous line
            print(f"[{speaker}] {text}", end="", flush=True)
            self._block_start = elapsed
        self._last_speaker = speaker
        self._last_end = elapsed

    def close(self) -> None:
        if self._last_speaker is not None:
            print()


class _StreamReconciler:
    def __init__(self, speaker: str):
        self.speaker = speaker
        self._prev_window_text = ""
        self.segments: list[dict] = []

    def step(self, window_text: str, elapsed: float) -> str:
        window_text = window_text.strip()
        new_piece = _new_text(self._prev_window_text, window_text).strip()
        self._prev_window_text = window_text
        if new_piece:
            start = self.segments[-1]["end"] if self.segments else 0.0
            self.segments.append({"start": start, "end": elapsed, "text": new_piece, "speaker": self.speaker})
        return new_piece


def _transcribe_window(asr_model, window, batch_size: int, language: str | None) -> str:
    if not window.size:
        return ""
    result = asr_model.transcribe(window, batch_size=batch_size, language=language)
    return " ".join(seg["text"].strip() for seg in result["segments"]).strip()


def run(
    asr_model,
    sample_rate: int,
    mic_device: int,
    loopback_device: int,
    chunk_seconds: float,
    step_seconds: float,
    overlap_seconds: float,
    batch_size: int,
    language: str | None,
    pause_gap_seconds: float,
    max_block_seconds: float,
) -> list[dict]:
    handle = capture.start_capture(
        sample_rate, mic_device, loopback_device, max_buffer_seconds=chunk_seconds + overlap_seconds
    )

    mic_reconciler = _StreamReconciler("user")
    system_reconciler = _StreamReconciler("system")
    console = _ConsoleWriter(pause_gap_seconds, max_block_seconds)
    start_time = time.monotonic()
    stop_flag = threading.Event()

    def wait_for_enter():
        input("Live transcription running... press Enter to stop: ")
        stop_flag.set()

    threading.Thread(target=wait_for_enter, daemon=True).start()

    print("Listening...")
    while not stop_flag.wait(step_seconds):
        elapsed = time.monotonic() - start_time
        mic_window, system_window = handle.read_last_seconds(chunk_seconds)

        mic_text = _transcribe_window(asr_model, mic_window, batch_size, language)
        new_piece = mic_reconciler.step(mic_text, elapsed)
        console.emit("user", new_piece, elapsed)

        system_text = _transcribe_window(asr_model, system_window, batch_size, language)
        new_piece = system_reconciler.step(system_text, elapsed)
        console.emit("system", new_piece, elapsed)

    console.close()
    handle.stop()
    return sorted(mic_reconciler.segments + system_reconciler.segments, key=lambda s: s["start"])
