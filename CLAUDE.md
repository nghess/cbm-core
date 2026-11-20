# cbm-core

Shared library for the cbm analysis projects. Imported by cbm-classify, cbm-decode, cbm-dimreduction, cbm-emd,
cbm-hmm, cbm-info, cbm-qc, cbm-remapping, cbm-sniff and cbm-sr (Nov 2026). The viewers, cbm-task-behavior and
cbm-event-locking don't import it; they keep their own copies of the definitions.
@README.md

## Rules for changing it

- A change here changes every project's results. Before changing a definition (epochs, box, bins, unit
  selection), say which projects and outputs it affects and get the user's go-ahead.
- Keep it pure numpy / pandas / scipy (scipy imported inside the functions that need it). No matplotlib,
  PySide or project-specific settings.
- Must work with pandas 2.x (sldata env) and 3.x (cbm-viewer's .venv). Run `pytest` in the sldata env
  (`C:/ProgramData/anaconda3/envs/sldata/python.exe -m pytest`); the `data` tests read E:/ and skip without it.
- Keep README.md's "Definitions and why" current: it is where other projects learn why a definition is
  what it is.
- Write numbers measured on the data into docstrings with their date (e.g. "309 sessions, Oct 2026").
