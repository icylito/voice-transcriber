# Future direction: a Windows GUI app

Everything in this repo today (and everything added by the record/URL/live-mode work) is CLI-only:
`transcribe.py` plus flags, with `config.toml` as the only place to change behavior. The intended
next evolution of this tool is a **Windows GUI application** built on top of the same pipeline —
not scheduled, not started, just the direction this project is expected to grow toward so it isn't
lost or re-derived later.

The GUI's job is to turn everything currently in `config.toml` (plus what this plan added) into
real, discoverable UI instead of a text file the user has to hand-edit:

- **Microphone and system-audio (loopback) device selection.** Today: `transcribe.py --list-devices`
  plus `record.mic_device` / `record.loopback_device` indices in `config.toml`. In the GUI: a
  device picker (dropdowns or a live level-meter view) instead of looking up an index and typing it
  into a config file.
- **Shortcut keys.** Today: none — record/live modes stop on a plain `input()` Enter-press in the
  terminal. In the GUI: configurable global hotkeys (start/stop recording, start/stop live mode)
  that work without the window being focused.
- **Model import / management.** A way to point the app at custom or additional model weights
  beyond the two built-in defaults, and manage what's downloaded locally.
- **Per-task model setup.** Today: a fixed mode→model default (`large-v3` for batch/record/URL,
  `large-v3-turbo` for live) with a single `--model quality|fast` override. In the GUI: each task
  type (live, recording, YouTube-pull, batch) gets its own independent quality-vs-speed choice,
  not just one global override.
- **Output/save location.** Today: `paths.videos_dir` / `paths.transcripts_dir` in `config.toml`.
  In the GUI: folder pickers, and per-job or per-task overrides instead of one fixed location for
  everything.
- **Everything else already in `config.toml`** — pause/block grouping thresholds
  (`pause_gap_seconds`, `max_block_seconds`), batch size, compute type, device, language,
  live-mode chunk/step/overlap timing — exposed as real settings UI, not text-file fields.
- **GUI Looks** basically going to have a minimal ui with 1 main view:
  1. 1 button for record/stop (after record was pressed) the other is for live recording 
  2. a voice wave bar only appearing as the voice is being recorded to help the user know 
    there is an actual input. Once the recording is doen it'd show the transcript in the app itself
    and it'll have a copy saved in the transcript/transcripts folder 
  3. live recording basically any input taken:
    (user)
    (system)
    is going to be displayed there for the audio captured if no audio captured from (user) then don't display user
    if no audio was captured from system then don't display system and if no audio captured at all then display nothing 
    no time stamps no audio no users/system, etc
  4. api options and model swapping on the spot
  5. keybinds/shortcuts 
  6. modern look something similar to ollama.com overall frontend design

This is explicitly not scheduled and nothing here commits the current CLI work to starting it —
it's a marker of intended direction, kept separate from CLAUDE.md (which documents what exists
today) and from any implementation plan (which is scoped to concrete, in-progress work).
