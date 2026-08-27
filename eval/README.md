# RAGAS evaluation harness

This folder contains a standalone evaluation runner for your existing agent.
The agent returns `message_ids` alongside `answer_text` on the pretty-answer path so eval can rebuild contexts; Telegram behavior for users is unchanged (still text-only when `answer_text` is set).

## What it does

- reads test cases from JSONL
- calls your current agent for each question
- collects:
  - `answer` (from `answer_text`)
  - `contexts` (full text restored from `message_ids` returned by the agent after rerank / pretty-answer path)

### Fixed RAGAS judge (not tied to `AI_MODEL`)

Metrics always call OpenRouter with **`OPENROUTER_API_KEY`**. The judge LLM is **not** inferred from the agent under test.

Set explicitly (recommended):

- `RAGAS_JUDGE_MODEL` — full OpenRouter slug, e.g. `openai/gpt-4o-mini`
- or `RAGAS_OPENROUTER_JUDGE_MODEL` — fallback if `RAGAS_JUDGE_MODEL` is empty (default in code: `google/gemini-2.0-flash-001`)

Changing `AI_PROVIDER` / `AI_MODEL` for the agent does **not** change the judge anymore.

Optional:

- `RAGAS_EMBEDDING_MODEL` — default `openai/text-embedding-3-small`
- runs RAGAS metrics
- saves detailed artifacts to `eval/results/`

## Install eval dependencies

```powershell
pip install -r requirements.eval.txt
```

## Dataset format

Input file: JSONL, one test case per line.

Required fields:

- `question` - user question
- `chat_id` - chat collection id in Qdrant

Optional fields:

- `id` - case id
- `ground_truth` - reference answer for stronger evaluation

Example:

```json
{"id":"case-1","chat_id":123456,"question":"Что обсуждали про релиз в апреле?","ground_truth":"В апреле обсуждали ..."}
```

See `eval/dataset.sample.jsonl`.

## Run (local)

```powershell
python -m eval.run_ragas_eval --dataset eval/dataset.sample.jsonl --output-dir eval/results
```

Optional flags:

- `--max-samples 20` - run only first N samples
- `--fail-on-thresholds eval/thresholds.example.json` - fail process when metrics are lower than thresholds
- `--from-predictions eval/results/predictions.jsonl` - skip the agent; run only RAGAS on saved predictions
- `--include-empty-answers` - по умолчанию строки с пустым `answer` **не** попадают в RAGAS; этим флагом можно вернуть старое поведение

### Filling `contexts` in eval

The runner restores contexts via Qdrant using **`message_ids`** from `agent.run()`.

After routing to **text** (`pretty_answer`), the agent still returns **`message_ids`** — the reranked ids used to build the pretty prompt (`cited_message_ids` internally). The Telegram bot keeps sending only `answer_text` when present; ids are for tooling/eval.

Re-run predictions after deploying this behavior so `predictions.jsonl` gets non-empty `contexts` when retrieval succeeds.

## Outputs

- `predictions.jsonl` - per-case generated answer and contexts
- `ragas_summary.json` - aggregate metrics

## CI usage

Use a small smoke dataset on each PR and full dataset nightly.
If you pass thresholds file, the script exits with non-zero code on quality regression.

## Docker run

An isolated eval service is added in `docker-compose.yml` under profile `eval`.
It does not affect normal `docker compose up` flow.

Run with sample dataset:

```powershell
docker compose --profile eval run --rm rag_eval
```

Run with your dataset:

```powershell
docker compose --profile eval run --rm rag_eval sh -lc "pip install --no-cache-dir -r requirements.eval.txt && python -m eval.run_ragas_eval --dataset eval/my_dataset.jsonl --output-dir eval/results --fail-on-thresholds eval/thresholds.example.json"
```

## About thresholds

`eval/thresholds.example.json` contains starter values, not RAGAS defaults.
RAGAS does not define universal pass thresholds because acceptable values depend on your domain, context quality, and response style.

Recommended approach:

- start with these values as smoke gate
- run on your historical real queries
- tune each threshold based on baseline median and business tolerance for failures
