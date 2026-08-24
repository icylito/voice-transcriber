"""CI check: confirms config.toml has every key transcribe.py's module-level constants read.

Stubs out the heavy ML/audio dependencies (whisperx, torch, capture, fetch, live), since CI
doesn't install them. Importing transcribe.py is what actually exercises every CONFIG[...] read,
so a successful import here means config.toml's schema is complete."""

import sys
import types
from pathlib import Path

for name in ("whisperx", "torch", "capture", "fetch", "live"):
    sys.modules.setdefault(name, types.ModuleType(name))

sys.path.insert(0, str(Path(__file__).parent.parent))
import transcribe  # noqa: E402

print(f"config.toml OK: {len(transcribe.CONFIG)} top-level sections loaded")
