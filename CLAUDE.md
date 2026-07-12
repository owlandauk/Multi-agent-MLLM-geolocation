# Claude Code Notes

This repository uses `AGENTS.md` as the canonical agent instruction file. Read
that first for project structure, commands, coding style, testing guidance, and
security rules.

Current experiment memory lives in:

- `geo_pipeline/results/v6b_diagnosis_notes.md`

Most recent diagnosis summary: `v6b_full.json` is bottlenecked by country-level
North America bias, especially United States false positives. The v7 changes
were designed to reduce early country lock-in, preserve top country candidates
during city/street reasoning, parse wrapper cue responses, and avoid geocoding
child locations with a conflicting parent country.
