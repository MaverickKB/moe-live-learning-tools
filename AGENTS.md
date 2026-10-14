# AGENTS.md (index)

This repository is a collection of independent reference tools. Each tool is
self-contained: its own `pyproject.toml`, `src/`, `tests/`, `README.md`, `AGENTS.md` and
`LESSONS.md`. There is no shared code yet; add a `common/` folder only for code that two
or more tools actually use.

| Tool | Agent guide |
|---|---|
| resident-trainer | [tools/resident-trainer/AGENTS.md](tools/resident-trainer/AGENTS.md) |

Rules for every tool:
- Run that tool's `pytest` from its directory before and after a change. Tests must pass on
  CPU without GPUs, servers or network.
- No user-specific paths, hosts, IPs or usernames. Use environment variables with generic
  defaults. Fixtures use generic placeholders (`host-a.example`, `atlas`, ...).
- Keep README claims to what was measured; mark everything else untested.
