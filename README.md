# voice-transcriber

Transcribes video and audio into timestamped Markdown, entirely on your machine. Drop a file in, record your mic and system audio, paste a URL, or run it live while you talk. Nothing leaves your computer. No cloud API, no server.

[![CI](https://github.com/icylito/voice-transcriber/actions/workflows/ci.yml/badge.svg)](https://github.com/icylito/voice-transcriber/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-BSD--2--Clause-blue)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows-0078D6)](#install)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](#install)

It runs on [WhisperX](https://github.com/m-bain/whisperX) by [m-bain](https://github.com/m-bain), which does the actual speech recognition (`faster-whisper`) and word-level alignment (wav2vec2). WhisperX installs as a standard pip dependency through requirements.txt, so no manual setup is needed. This tool wraps that engine with what WhisperX doesn't handle on its own: capturing a mic and system audio at once, pulling audio from a URL, running live, and cleaning up the output.

![Terminal screenshot of transcribe.py's usage help text](image.png)

### A new run

This is captured output, not a mockup. `voice --url <a short speech>` with the quality model:

```
Loading Whisper model (large-v3) on cuda...
[AddressonVietnamWarProtests.wav] loading audio
[AddressonVietnamWarProtests] transcribing
  transcribed 5 segments (language=en)
loading alignment model for language=en
[AddressonVietnamWarProtests] restoring punctuation for 14 segments via qwen3:4b-instruct-2507-q4_K_M...
What do you want the file name to be? [AddressonVietnamWarProtests]:
[AddressonVietnamWarProtests] wrote transcripts\AddressonVietnamWarProtests.md
```

And the transcript it wrote. Every block gets a timestamp range, punctuation already cleaned up:

```markdown
# AddressonVietnamWarProtests

**[00:00 - 00:37]** Sometimes it appears that we're reaching a period when our senses and our
minds will no longer respond to moderate stimulation. We seem to be approaching an age of the
gross. Persuasion through speeches and books is too often discarded for disruptive demonstrations
aimed at bludgeoning the unconvinced into action...

**[00:38 - 01:09]** At the zenith of physical power and sensitivity overwhelm themselves with
drugs and artificial stimulants. Subtlety is lost and fine distinctions based on acute reasoning
are carelessly ignored in a headlong jump to a predetermined conclusion...
```

There's no graphical interface yet, so these are terminal captures instead of app screenshots. A GUI is a planned direction, not built. See `future_plan.md`.

## Features

- Drop video or audio files into `videos/`, run `voice`, get transcripts back.
- `--record` captures your mic and whatever's playing through your speakers, separately, so the transcript labels who said what.
- `--url <URL>` pulls audio from a link through `yt-dlp` and transcribes it.
- `--live` does the same dual capture, streamed to your console as you speak.
- `--pick` lets you choose which pending video to transcribe.
- Two optional extras run through a local LLM in Ollama: punctuation cleanup, and a title generated from what's in the recording instead of a generic filename. Both are on by default. If Ollama isn't running, transcription still finishes, it skips those two steps.

Every run writes a timestamped `.md` file to `transcripts/`. Turn on `.srt` output too if you want subtitles.

## Install

You need Windows, Python 3.11 or newer, and `ffmpeg` on your PATH (`winget install Gyan.FFmpeg`, then open a fresh terminal). An NVIDIA GPU with CUDA changes this from usable to fast. See Benchmarks below for the difference. Ollama is optional, only needed for the punctuation and title extras.

```bash
git clone https://github.com/icylito/voice-transcriber.git
cd voice-transcriber

python -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install -r requirements.txt

ollama pull qwen3:4b-instruct-2507-q4_K_M
```

Install torch before whisperx, in that order. Installing whisperx first pulls the default CPU-only torch wheel on Windows, and you lose GPU acceleration with nothing telling you it happened.

Check your GPU's VRAM with `nvidia-smi` before your first real run. Configuration below covers what to change if your card is smaller than the one this was built on.

## Usage

```bash
voice                          # process every new file in videos/
voice path\to\one.mp4          # transcribe a single file
voice --pick                   # interactively choose one pending video
voice --record                 # record mic+system audio, transcribe on stop
voice --url <URL>              # download audio via yt-dlp, then transcribe it
voice --live                   # live chunked transcription, streamed to console
voice --list                   # show pending/done videos, don't transcribe
voice --list-devices           # show mic/loopback device indices for config.toml
voice --model quality          # force the quality model regardless of mode
voice --model fast             # force the fast model regardless of mode
```

`voice` is `voice.bat`. It puts `ffmpeg` on PATH and runs `transcribe.py`. Add the repo root to your PATH and it works from anywhere, or run `.\voice` from inside the repo.

## Configuration

Everything tunable lives in `config.toml`. No code changes needed.

**Swap the Whisper model:**
```toml
[model.quality]
name = "large-v3"         # batch/--pick/single-file/--record/--url

[model.fast]
name = "large-v3-turbo"   # --live only
```
`large-v3` needs about 5.5GB of VRAM at `float16`. If your card has less, drop to a smaller Whisper variant, or switch `model.compute_type` to `int8` or `int8_float16` to shrink the footprint.

**Swap the local LLM** (punctuation cleanup and title generation):
```toml
[llm]
model = "qwen3:4b-instruct-2507-q4_K_M"
```
Pull whatever model you want with `ollama pull <name>` and set it here. Watch out for reasoning models: the plain `qwen3:4b` tag burns 30 to 40 seconds thinking per call with no quality gain on this task, which is why this project pulls the `-instruct` tag specifically.

**Turn the LLM extras off:**
```toml
[llm]
enabled = true      # punctuation and capitalization cleanup
auto_title = true   # LLM-generated transcript titles
auto_start = true   # offer to launch Ollama if it's not running
```
Each toggles independently. Turn both `enabled` and `auto_title` off and the tool never touches Ollama.

## Benchmarks

Two short public-domain speeches from the [Internet Archive](https://archive.org/details/Greatest_Speeches_of_the_20th_Century), run through `voice --url <clip>` end to end: download, model load, transcribe, align, punctuation cleanup. The command someone would type, not isolated inference time.

| Clip | Length | Model | Wall clock | Real-time factor |
|---|---|---|---|---|
| Address on Vietnam War Protests | 2:16 | large-v3 (quality) | 44.0s | 3.1x faster than real time |
| Address on Vietnam War Protests | 2:16 | large-v3-turbo (fast) | 26.8s | 5.1x faster than real time |
| Abdication Address | 6:42 | large-v3 (quality) | 2m 48s | 2.4x faster than real time |
| Abdication Address | 6:42 | large-v3-turbo (fast) | 35.5s | 11.3x faster than real time |

These numbers belong to the hardware below. GPU compute, VRAM, CPU, and disk speed all move them. Forced alignment scales with how many speech segments the audio has, not with length alone, so a longer clip with fewer, longer segments can align faster than a short one full of rapid back and forth lines.

### Hardware these ran on

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3080, 10GB VRAM |
| CPU | Intel Core i9-12900K |
| RAM | 32GB DDR5 6000MHz CL30 |
| Storage | NVMe SSD |
| OS | Windows 11 Pro |

## Contributing

CI (`.github/workflows/ci.yml`) runs syntax checks, `ruff`, config validation, and `pytest` on every push, but it only covers pure logic (`tests/`), not the actual GPU pipeline or real audio capture, since GitHub's runners don't have a GPU or a mic. If you're changing the pipeline (`transcribe.py`, `capture.py`, `live.py`, `fetch.py`), run the relevant mode end to end and check the transcript it produces before opening a PR, CI passing isn't enough on its own. `CLAUDE.md` covers the architecture and the reasoning behind the less obvious decisions, read it first if you're touching capture or the LLM steps.

Open an issue for bugs or feature ideas. For anything Windows-audio related, include your device names from `voice --list-devices`.

## License

BSD 2-Clause, see [LICENSE](LICENSE).

This project runs on [WhisperX](https://github.com/m-bain/whisperX), also BSD 2-Clause, copyright Max Bain. Full credit to that project for the speech recognition and alignment engine underneath this tool.

This project was fully written by Claude [claude.ai](https://claude.ai/)
