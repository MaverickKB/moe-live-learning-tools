# moe-live-learning-tools

Reference extracts from a research project on teaching a large mixture-of-experts model
new facts **while it keeps serving**. Each tool lives in `tools/<name>/` with its own
README, agent guide, lessons and tests.

> These are **extracts and engineering examples, not maintained products.** They ran on
> one specific two-machine setup and have not been run anywhere else. They are shared so
> that people and their coding agents can learn from them and adapt them.

## Tools

| Tool | What it is | Status |
|---|---|---|
| [`resident-trainer`](tools/resident-trainer/) | Trains a few added shared-expert neurons at one middle layer inside a live tensor-parallel server, with paired job start, trainer-only memory floor, frozen-weight proof and no-restart code loading | Ran live on the origin setup (16/24 dev strict, 11/24 rewordings, from 0; controls unchanged). Engine glue is reference-only. CPU tests pass. Untested on any other hardware or model. |

## Quick start

```bash
cd tools/resident-trainer
python -m venv .venv && . .venv/bin/activate
pip install -e '.[test]'
pytest
```

## License

MIT for the original code, with Apache-2.0 portions derived from upstream projects. See
`LICENSE`, `NOTICE` and `licenses/`.
