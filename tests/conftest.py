"""Stubs whisperx/torch/capture/fetch before any test imports transcribe or live, since CI
doesn't install the GPU/audio stack. Only pure logic gets tested here, not the real ML pipeline.

live isn't stubbed. test_live.py needs the real module, and live.py's own `import capture`
resolves to the stub above regardless of which test imports it first."""

import sys
import types
from pathlib import Path

for name in ("whisperx", "torch", "capture", "fetch"):
    sys.modules.setdefault(name, types.ModuleType(name))

sys.path.insert(0, str(Path(__file__).parent.parent))
