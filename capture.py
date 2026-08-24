"""Mic + system-audio (WASAPI loopback) capture, kept as two separate streams.

Both streams stay separate rather than being mixed down, so the transcription pipeline can
label segments (user) vs (system) instead of transcribing an anonymous blend.

The two streams use different backends:

- **Loopback (system audio)** goes through PyAudioWPatch/WASAPI, which works reliably here.
  Exclusive-mode streams can't loopback, so this opens a shared-mode input stream on the
  loopback analogue of the default output device.
- **Mic** goes through an `ffmpeg` subprocess (DirectShow input), not PyAudioWPatch. Confirmed by
  direct testing: PortAudio (both PyAudioWPatch and a separate vanilla `sounddevice` install) fails
  to open this machine's USB mic under every host API (WASAPI/MME/WDM-KS) with generic "Invalid
  device"/"Unanticipated host error" codes, even though the exact same device opens fine through
  Windows' own recorder and through `ffmpeg -f dshow`. Since `ffmpeg` is already a hard dependency
  of this project, routing mic capture through it sidesteps the PortAudio-specific incompatibility
  entirely rather than working around it inside PortAudio.
"""

import subprocess
import threading
import time

import numpy as np
import pyaudiowpatch as pyaudio
from scipy.signal import resample_poly

FRAMES_PER_BUFFER = 1024
FFMPEG_MIC_CHUNK_BYTES = 2048  # ~1024 mono int16 samples per read


def _resample(mono: np.ndarray, native_rate: int, target_rate: int) -> np.ndarray:
    if mono.size == 0 or native_rate == target_rate:
        return mono.astype(np.float32)
    return resample_poly(mono, target_rate, native_rate).astype(np.float32)


class _StreamBuffer:
    """Thread-safe accumulator for one audio stream, downmixed to mono at its native rate.

    When max_native_samples is set, old audio is dropped as new audio arrives, keeping reads
    cheap regardless of session length (needed for live mode's per-step reads). Record mode
    leaves it unbounded since it only reads once, on stop.
    """

    def __init__(self, native_rate: int, channels: int, target_rate: int, max_native_samples: int | None = None):
        self._native_rate = native_rate
        self._channels = channels
        self._target_rate = target_rate
        self._max_native_samples = max_native_samples
        self._chunks: list[np.ndarray] = []
        self._total_samples = 0
        self._lock = threading.Lock()

    def write(self, raw_bytes: bytes) -> None:
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        samples = samples.reshape(-1, self._channels).mean(axis=1)
        with self._lock:
            self._chunks.append(samples)
            self._total_samples += samples.size
            if self._max_native_samples is not None:
                while self._total_samples > self._max_native_samples and len(self._chunks) > 1:
                    dropped = self._chunks.pop(0)
                    self._total_samples -= dropped.size

    def _snapshot(self) -> np.ndarray:
        with self._lock:
            return np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype=np.float32)

    def read_all(self) -> np.ndarray:
        return _resample(self._snapshot(), self._native_rate, self._target_rate)

    def read_last_seconds(self, seconds: float) -> np.ndarray:
        mono = self._snapshot()
        needed = int(seconds * self._native_rate)
        tail = mono[-needed:] if needed and mono.size > needed else mono
        return _resample(tail, self._native_rate, self._target_rate)


class CaptureHandle:
    def __init__(
        self,
        pa: pyaudio.PyAudio,
        mic_proc: subprocess.Popen,
        loopback_stream,
        mic_buffer: _StreamBuffer,
        loopback_buffer: _StreamBuffer,
    ):
        self._pa = pa
        self._mic_proc = mic_proc
        self._loopback_stream = loopback_stream
        self._mic_buffer = mic_buffer
        self._loopback_buffer = loopback_buffer

    def read_last_seconds(self, seconds: float) -> tuple[np.ndarray, np.ndarray]:
        return self._mic_buffer.read_last_seconds(seconds), self._loopback_buffer.read_last_seconds(seconds)

    def _stop_mic(self) -> None:
        try:
            self._mic_proc.stdin.write(b"q")
            self._mic_proc.stdin.flush()
            self._mic_proc.wait(timeout=2)
        except Exception:
            self._mic_proc.kill()
            try:
                self._mic_proc.wait(timeout=5)
            except Exception:
                pass

    def stop(self) -> tuple[np.ndarray, np.ndarray]:
        self._stop_mic()
        self._loopback_stream.stop_stream()
        self._loopback_stream.close()
        mic_audio = self._mic_buffer.read_all()
        system_audio = self._loopback_buffer.read_all()
        self._pa.terminate()
        return mic_audio, system_audio


def list_devices() -> None:
    p = pyaudio.PyAudio()
    try:
        print("Capture devices (use the index for record.mic_device / record.loopback_device in config.toml):")
        for d in p.get_device_info_generator():
            if d["maxInputChannels"] <= 0:
                continue
            kind = "loopback" if d.get("isLoopbackDevice") else "mic"
            print(f"  [{d['index']}] ({kind}) {d['name']} @ {int(d['defaultSampleRate'])}Hz")
    finally:
        p.terminate()


def _resolve_mic(p: pyaudio.PyAudio, device_index: int) -> dict:
    if device_index == -1:
        # Use the WASAPI-scoped default, not get_default_input_device_info() (which can resolve
        # to a different host API's duplicate device entry for the same hardware, e.g. MME/
        # DirectSound). Only the friendly name is actually used (ffmpeg's dshow input takes a
        # device name, not a PortAudio index) but resolving through WASAPI keeps this consistent
        # with --list-devices, which enumerates via the same host API.
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        return p.get_device_info_by_index(wasapi_info["defaultInputDevice"])
    return p.get_device_info_by_index(device_index)


def _resolve_loopback(p: pyaudio.PyAudio, device_index: int) -> dict:
    if device_index == -1:
        return p.get_default_wasapi_loopback()
    return p.get_device_info_by_index(device_index)


def _start_ffmpeg_mic_capture(device_name: str, sample_rate: int, buffer: _StreamBuffer) -> subprocess.Popen:
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-f",
            "dshow",
            "-i",
            f"audio={device_name}",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "s16le",
            "-loglevel",
            "error",
            "-",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    time.sleep(0.3)  # give ffmpeg a moment to fail fast on a bad/unavailable device
    if proc.poll() is not None:
        stderr = proc.stderr.read().decode(errors="replace").strip()
        raise RuntimeError(f"ffmpeg couldn't open mic '{device_name}': {stderr}")

    def pump():
        while True:
            data = proc.stdout.read(FFMPEG_MIC_CHUNK_BYTES)
            if not data:
                break
            buffer.write(data)

    threading.Thread(target=pump, daemon=True).start()
    return proc


def start_capture(
    sample_rate: int, mic_device: int, loopback_device: int, max_buffer_seconds: float | None = None
) -> CaptureHandle:
    p = pyaudio.PyAudio()
    mic_info = _resolve_mic(p, mic_device)
    loopback_info = _resolve_loopback(p, loopback_device)

    loopback_channels = int(loopback_info["maxInputChannels"])
    loopback_native_rate = int(loopback_info["defaultSampleRate"])
    max_loopback_samples = int(max_buffer_seconds * loopback_native_rate) if max_buffer_seconds else None
    loopback_buffer = _StreamBuffer(loopback_native_rate, loopback_channels, sample_rate, max_loopback_samples)

    def loopback_callback(in_data, frame_count, time_info, status):
        loopback_buffer.write(in_data)
        return (None, pyaudio.paContinue)

    loopback_stream = p.open(
        format=pyaudio.paInt16,
        channels=loopback_channels,
        rate=loopback_native_rate,
        input=True,
        input_device_index=loopback_info["index"],
        frames_per_buffer=FRAMES_PER_BUFFER,
        stream_callback=loopback_callback,
    )

    # ffmpeg is asked to resample/downmix to sample_rate/mono itself, so native_rate == target_rate
    # here and _StreamBuffer's resample step is a no-op for the mic path.
    max_mic_samples = int(max_buffer_seconds * sample_rate) if max_buffer_seconds else None
    mic_buffer = _StreamBuffer(sample_rate, 1, sample_rate, max_mic_samples)
    mic_proc = _start_ffmpeg_mic_capture(mic_info["name"], sample_rate, mic_buffer)

    return CaptureHandle(p, mic_proc, loopback_stream, mic_buffer, loopback_buffer)


def record_until_enter(sample_rate: int, mic_device: int, loopback_device: int) -> tuple[np.ndarray, np.ndarray]:
    handle = start_capture(sample_rate, mic_device, loopback_device)
    input("Recording... press Enter to stop: ")
    return handle.stop()
