---
description: Analyze the diff and create a well-formed commit
allowed-tools: Bash(git *), Read, Grep, Glob
---

Create a git commit for the current changes:

1. Run `git status` and `git diff` (plus `git diff --staged`) to see everything pending.
2. Sanity-check: nothing under `data/`, no `*.csv`/`*.zip`/model weights, no `kaggle.json` or other secrets. If any are staged, stop and tell me instead of committing.
3. If the changes contain clearly unrelated work, propose splitting into multiple commits and ask which grouping I want.
4. Stage the relevant files and commit with a message that has:
   - An imperative-mood subject line under 72 chars (e.g. "Add snap-aligned frame windowing to data pipeline")
   - A short body explaining *why*, when the diff alone doesn't make it obvious
5. Show me the final `git log -1 --stat`.

$ARGUMENTS
