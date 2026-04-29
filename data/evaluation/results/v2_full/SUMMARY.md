# RAGAS evaluation — testset_v2 (data-grounded references)

Run date: 2026-04-29
Testset: `data/evaluation/testset_v2.json` (25 questions, references regenerated from Neo4j Aura + ChromaDB).
LLM (system + judge): defaults per mode (classic=`gemini-2.5-flash`, agent=`gemini-2.5-flash-lite`, multi supervisor=`gemini-2.5-pro` + sub-agents=`gemini-2.5-flash-lite`). RAGAS judge: `gemini-2.5-flash`.
API: local FastAPI on `127.0.0.1:8000` against Neo4j Aura (11,900 nodes / 381,359 rels) and local ChromaDB (5,654 chunks).

## RAGAS metrics (n=25)

| Metric | Classic | Agent | Multi |
| --- | --- | --- | --- |
| AnswerCorrectness | 0.544 | **0.610** | 0.551 |
| AnswerRelevancy | 0.695 | **0.735** | 0.708 |
| ContextPrecision | **0.760** | 0.207 | 0.165 |
| ContextRecall | 0.504 | **0.692** | 0.431 |
| Faithfulness | **0.910** | 0.680 | 0.792 |
| latency_ms (avg) | 5,688 | 11,225 | 19,788 |
| total runtime (s) | 2,341 | 4,025 | ~3,800 |

## Agent tool selection (custom metric)

| Mode | Precision | Recall | F1 | Goal accuracy |
| --- | --- | --- | --- | --- |
| Agent | 0.269 | 0.960 | 0.383 | **100% (25/25)** |
| Multi | 0.000 | 0.000 | 0.000 | **100% (25/25)** |

Multi P/R/F1 = 0 is an artifact: the supervisor delegates via `ask_*_expert` wrappers; the underlying tool calls live inside sub-agents and are not exposed to the eval harness. Goal accuracy still tracks correctly.

## vs prior baseline (n=3, 2026-04-21)

| Metric | Old Classic (n=3) | New Classic (n=25) | Δ |
| --- | --- | --- | --- |
| AnswerCorrectness | 0.49 | 0.544 | +0.054 |
| AnswerRelevancy | 0.84 | 0.695 | -0.145 |
| ContextPrecision | 0.83 | 0.760 | -0.07 |
| ContextRecall | **0.22** | **0.504** | **+0.284 (+130%)** |
| Faithfulness | 0.94 | 0.910 | -0.03 |

Old baseline ran only 3 questions (`--limit 3`), so deltas mix two effects: data-grounded references (intended) and larger sample (more variance). The headline finding holds: **ContextRecall jumps from 0.22 to 0.504 once references match what the system actually retrieves.**

## Known issues

- **RAGAS judge timeouts**: ~10 `max_tokens exceeded` warnings across the three runs (q06, q08, q12, q15 are recurrent). These tank averages because failed metrics return NaN and the runner does not impute. Could be mitigated by raising `max_tokens` on the judge or by switching to `gemini-2.5-pro` for judging.
- **Warfarin has no `DrugCategory`** in the KG. Surfaced honestly in q13/q15 references rather than masked. Real ingestion gap: only `Factor Xa Inhibitor [EPC]` exists for apixaban/rivaroxaban; no `Anticoagulant`, `Coumarin`, or `Vitamin K Antagonist` category nodes were created.
- **Agent precision low (0.27)**: agent calls extra tools beyond the ground-truth set. Recall is high (0.96) and goal accuracy 100%, so user experience is unaffected; cost and latency are.

## Files

- `ragas_classic.csv`, `ragas_agent.csv`, `ragas_multi.csv` — per-sample RAGAS scores (gitignored, regenerable).
- `agent_tools_agent.csv`, `agent_tools_multi.csv` — per-sample tool selection metrics (gitignored).
- `classic_log.txt`, `agent_log.txt`, `multi_log.txt` — full execution logs (gitignored).
- This `SUMMARY.md` is the persistent record.

## Reproduce

```bash
# 1. Set env
export GEMINI_API_KEY=...
# .env.aura with Neo4j Aura credentials

# 2. Start API against Aura
NEO4J_URI=... NEO4J_USER=... NEO4J_PASSWORD=... \
  uv run uvicorn pharmagraphrag.api.main:app --host 127.0.0.1 --port 8000

# 3. Run each mode
uv run python scripts/run_evaluation.py --mode classic \
  --testset data/evaluation/testset_v2.json \
  --api-url http://127.0.0.1:8000 \
  --output-dir data/evaluation/results/v2_full
# Repeat with --mode agent and --mode multi
```
