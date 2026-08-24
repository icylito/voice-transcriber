"""yt-dlp wrapper: download a URL's audio as a WAV file for the transcription pipeline.

Downloads to a temp file rather than piping. yt-dlp can't combine -x (audio extraction) with
-o - (stdout piping), and piping raw video into another process is fragile (throttling, seeking,
format-detection issues), per yt-dlp issue #2627. Mirrors the temp-file pattern whisper.cpp's
own yt-wsp.sh script uses.
"""

import subprocess
import sys
from pathlib import Path


def download_audio(url: str, dest_dir: Path) -> Path:
    before = set(dest_dir.iterdir()) if dest_dir.exists() else set()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "yt_dlp",
            "-x",
            "--audio-format",
            "wav",
            "--no-playlist",
            "-o",
            str(dest_dir / "%(title)s.%(ext)s"),
            "--print",
            "after_move:filepath",
            url,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr.strip()}")

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"yt-dlp did not report an output path.\nstdout: {result.stdout}\nstderr: {result.stderr}")
    printed_path = Path(lines[-1])
    if printed_path.exists():
        return printed_path

    # yt-dlp's --print after_move:filepath doesn't always match what lands on disk on Windows.
    # A title with a character illegal in Windows filenames (e.g. "?") gets swapped for a
    # lookalike ("？") in the real filename, but --print reports the un-swapped name. Instead of
    # replicating yt-dlp's sanitization rules, check what new file appeared in dest_dir.
    after = set(dest_dir.iterdir())
    new_files = [p for p in (after - before) if p.suffix.lower() == printed_path.suffix.lower()]
    if len(new_files) == 1:
        return new_files[0]
    raise RuntimeError(
        f"yt-dlp reported '{printed_path}' but that file doesn't exist, and "
        f"{'no' if not new_files else 'more than one'} new matching file turned up in {dest_dir} "
        f"instead."
    )
