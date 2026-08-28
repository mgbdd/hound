import argparse
import json
import os
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    load_dotenv = None  # type: ignore[misc, assignment]

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate

# RAGAS 0.4+: `from ragas.metrics.collections import answer_relevancy` resolves to a
# *subpackage* named answer_relevancy, not a metric — evaluate() then raises:
# "All metrics must be initialised metric objects".
# Use the package's pre-built singleton instances (same as evaluate(metrics=None)).
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
from ragas.metrics._faithfulness import faithfulness

from agent.graph import build_agent
from RAG.RAG import RAG


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    Load eval records from JSONL **or** from multiple whitespace-separated JSON objects
    (pretty-printed blocks `{ ... }`), or from a single JSON array `[{...}, {...}]`.
    """
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []

    rows: list[dict[str, Any]] = []

    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON array in {path}: {exc}") from exc
        if not isinstance(parsed, list):
            raise ValueError(f"Expected JSON array at root in {path}")
        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                raise ValueError(f"Item #{i} in {path} is not an object")
            rows.append(item)
        return rows

    decoder = json.JSONDecoder()
    idx = 0
    obj_index = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON object starting near offset {idx} in {path}: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"Item #{obj_index} in {path} is not a JSON object")
        rows.append(obj)
        obj_index += 1
        idx = end

    return rows


def ensure_required_fields(rows: list[dict[str, Any]]) -> None:
    required = {"question", "chat_id"}
    for idx, row in enumerate(rows):
        missing = [k for k in required if k not in row]
        if missing:
            raise ValueError(f"Row #{idx} misses required fields: {missing}")


def ensure_predictions_for_ragas(rows: list[dict[str, Any]]) -> None:
    """Rows produced by run_predictions or loaded from predictions.jsonl."""
    for idx, row in enumerate(rows):
        if "question" not in row:
            raise ValueError(f"Prediction row #{idx} misses 'question'")
        if "contexts" not in row:
            raise ValueError(f"Prediction row #{idx} misses 'contexts'")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def restore_contexts(rag: RAG, chat_id: Any, message_ids: list[str]) -> list[str]:
    if not message_ids:
        return []
    merged_by_id = rag.fetch_merged_texts_for_message_ids(chat_id=chat_id, message_ids=message_ids)
    ordered_contexts: list[str] = []
    for mid in message_ids:
        key = str(mid).strip()
        text = merged_by_id.get(key, "")
        if text:
            ordered_contexts.append(text)
    return ordered_contexts


def run_predictions(
    rows: list[dict[str, Any]],
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    agent = build_agent()  # синглтон (lru_cache) — тот же экземпляр, что импортирован в agent.graph
    rag = agent.rag  # переиспользуем уже загруженные модели RAG
    out: list[dict[str, Any]] = []

    selected_rows = rows[:max_samples] if max_samples else rows
    for i, row in enumerate(selected_rows, start=1):
        chat_id = row["chat_id"]
        question = str(row["question"])
        case_id = row.get("id", f"case-{i}")

        response = agent.search(question, chat_id)
        message_ids = [str(x).strip() for x in (response.get("message_ids", []) or [])]
        answer = (response.get("answer_text", "") or "").strip()
        contexts = restore_contexts(rag=rag, chat_id=chat_id, message_ids=message_ids)

        out_row: dict[str, Any] = {
            "id": case_id,
            "chat_id": chat_id,
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "message_ids": message_ids,
        }
        if "ground_truth" in row:
            out_row["ground_truth"] = row["ground_truth"]

        out.append(out_row)
        print(f"[{i}/{len(selected_rows)}] done: {case_id}")

    return out


def _normalize_openrouter_judge_model(model: str) -> str:
    """
    OpenRouter expects slugs like ``openai/gpt-4o-mini`` or ``google/gemini-2.0-flash-001``.
    Values from ``AI_MODEL`` for Yandex / GigaChat / etc. are not valid there.
    """
    fallback = (
        os.getenv("RAGAS_OPENROUTER_JUDGE_MODEL") or "google/gemini-2.0-flash-001"
    ).strip()
    m = (model or "").strip()
    if not m:
        return fallback
    if "/" in m:
        return m
    ml = m.lower()
    if ml.startswith("gemini"):
        return f"google/{m}"
    if ml.startswith(("gpt-", "o1", "o3", "o4")):
        return f"openai/{m}"
    if ml.startswith("claude-"):
        return f"anthropic/{m}"
    if ml.startswith("mistral"):
        return f"mistralai/{m}"
    # YandexGPT / Alice / folder URIs — not OpenRouter IDs (400 invalid model).
    if "yandex" in ml or "alice" in ml or "gpt://" in ml or "gigachat" in ml:
        print(
            f"RAGAS judge: model {model!r} is not an OpenRouter slug; "
            f"using {fallback!r} (set RAGAS_JUDGE_MODEL or RAGAS_OPENROUTER_JUDGE_MODEL)."
        )
        return fallback
    print(
        f"RAGAS judge: model {model!r} has no provider prefix; "
        f"using {fallback!r} (set RAGAS_JUDGE_MODEL to full OpenRouter id)."
    )
    return fallback


def _resolve_openrouter_config() -> tuple[str, str, str, str]:
    """api_key, base_url, judge_model, embedding_model.

    Судья RAGAS **не** берётся из AI_MODEL (тестируемая модель агента), чтобы при смене провайдера
    метрики оставались сопоставимыми. Задайте явно RAGAS_JUDGE_MODEL или RAGAS_OPENROUTER_JUDGE_MODEL.
    """
    try:
        from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL
    except ImportError:
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")

    api_key = (OPENROUTER_API_KEY or "").strip()
    if not api_key:
        raise RuntimeError(
            "Для RAGAS через OpenRouter задайте OPENROUTER_API_KEY в .env "
            "(или выставьте переменную окружения при запуске контейнера)."
        )
    base_url = (OPENROUTER_BASE_URL or "https://openrouter.ai/api/v1").rstrip("/")

    judge_raw = (os.getenv("RAGAS_JUDGE_MODEL") or "").strip()
    if not judge_raw:
        judge_raw = (os.getenv("RAGAS_OPENROUTER_JUDGE_MODEL") or "google/gemini-2.0-flash-001").strip()
    judge_model = _normalize_openrouter_judge_model(judge_raw)

    embedding_model = (os.getenv("RAGAS_EMBEDDING_MODEL") or "openai/text-embedding-3-small").strip()

    return api_key, base_url, judge_model, embedding_model


def _build_openrouter_ragas_llm_and_embeddings() -> tuple[ChatOpenAI, OpenAIEmbeddings]:
    api_key, base_url, judge_model, embedding_model = _resolve_openrouter_config()

    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=judge_model,
        temperature=0,
        max_retries=2,
    )

    try:
        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=api_key,
            base_url=base_url,
        )
    except TypeError:
        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=api_key,
            openai_api_base=base_url,
        )

    print(
        f"RAGAS judge: OpenRouter model={judge_model!r}, "
        f"embeddings={embedding_model!r}, base_url={base_url!r}"
    )
    return llm, embeddings


def _evaluation_result_to_scores(raw: Any) -> dict[str, float]:
    """RAGAS 0.4+ ``evaluate()`` returns ``EvaluationResult`` with per-row ``scores`` list."""
    from ragas.dataset_schema import EvaluationResult
    from ragas.utils import safe_nanmean

    if isinstance(raw, dict):
        return {str(k): float(v) for k, v in raw.items() if isinstance(v, (int, float))}

    if isinstance(raw, EvaluationResult):
        rows = raw.scores
        if not rows:
            return {}
        out: dict[str, float] = {}
        for key in rows[0].keys():
            vals: list[float] = []
            for row in rows:
                if not isinstance(row, dict) or key not in row:
                    continue
                v = row[key]
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            if not vals:
                continue
            agg = safe_nanmean(vals)
            if isinstance(agg, float) and agg == agg:  # skip NaN
                out[str(key)] = float(agg)
        return out

    raise TypeError(f"Unexpected evaluate() return type: {type(raw)!r}")


def _answer_for_ragas(row: dict[str, Any]) -> str:
    if "answer" in row:
        return str(row.get("answer", "") or "")
    return str(row.get("answer_text", "") or "")


def exclude_empty_answer_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Строки без текста ответа не участвуют в RAGAS (иначе метрики искажаются)."""
    kept: list[dict[str, Any]] = []
    skipped = 0
    for row in rows:
        if _answer_for_ragas(row).strip():
            kept.append(row)
        else:
            skipped += 1
    return kept, skipped


def build_ragas_dataset(rows: list[dict[str, Any]]) -> Dataset:
    has_ground_truth = any("ground_truth" in row for row in rows)
    data: dict[str, list[Any]] = {
        "question": [row["question"] for row in rows],
        "answer": [_answer_for_ragas(row) for row in rows],
        "contexts": [row.get("contexts", []) for row in rows],
    }
    if has_ground_truth:
        data["ground_truth"] = [row.get("ground_truth", "") for row in rows]
    return Dataset.from_dict(data)


def run_ragas(rows: list[dict[str, Any]]) -> dict[str, float]:
    dataset = build_ragas_dataset(rows)
    llm, embeddings = _build_openrouter_ragas_llm_and_embeddings()
    raw = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )
    return _evaluation_result_to_scores(raw)


def load_thresholds(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Thresholds file must be a JSON object")
    return {str(k): float(v) for k, v in raw.items()}


def check_thresholds(summary: dict[str, float], thresholds: dict[str, float]) -> list[str]:
    failures: list[str] = []
    for metric, expected_min in thresholds.items():
        actual = summary.get(metric)
        if actual is None:
            failures.append(f"{metric}: missing in summary")
            continue
        if actual < expected_min:
            failures.append(f"{metric}: {actual:.4f} < {expected_min:.4f}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone RAGAS evaluation for current agent")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Input JSONL with eval questions (required unless --from-predictions)",
    )
    parser.add_argument(
        "--from-predictions",
        type=Path,
        default=None,
        help="Skip agent run: load predictions JSONL (e.g. eval/results/predictions.jsonl) and run only RAGAS",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("eval/results"), help="Output artifacts directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Evaluate first N samples only")
    parser.add_argument(
        "--fail-on-thresholds",
        type=Path,
        default=None,
        help="Path to JSON with metric minimum thresholds",
    )
    parser.add_argument(
        "--include-empty-answers",
        action="store_true",
        help="Учитывать в RAGAS и строки с пустым ответом (по умолчанию они исключаются)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.from_predictions is None and args.dataset is None:
        raise SystemExit("Specify --dataset or --from-predictions PATH")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.from_predictions is not None:
        predictions = load_jsonl(args.from_predictions)
        if args.max_samples:
            predictions = predictions[: args.max_samples]
        ensure_predictions_for_ragas(predictions)
        print(f"Loaded {len(predictions)} rows from {args.from_predictions} (RAGAS only, agent skipped).")
    else:
        assert args.dataset is not None
        rows = load_jsonl(args.dataset)
        ensure_required_fields(rows)

        predictions = run_predictions(rows=rows, max_samples=args.max_samples)
        predictions_path = args.output_dir / "predictions.jsonl"
        write_jsonl(predictions_path, predictions)
        print(f"Predictions saved to: {predictions_path}")

    ragas_input = predictions
    if not args.include_empty_answers:
        ragas_input, n_skip_empty = exclude_empty_answer_rows(predictions)
        if n_skip_empty:
            print(
                f"RAGAS: исключено {n_skip_empty} строк с пустым ответом "
                f"(передайте --include-empty-answers, чтобы оценивать и их)."
            )
        if not ragas_input:
            print("Ошибка: после исключения пустых ответов не осталось строк для RAGAS.", flush=True)
            return 1

    summary = run_ragas(ragas_input)
    summary_path = args.output_dir / "ragas_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RAGAS summary saved to: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.fail_on_thresholds:
        thresholds = load_thresholds(args.fail_on_thresholds)
        failures = check_thresholds(summary=summary, thresholds=thresholds)
        if failures:
            print("Threshold check failed:")
            for line in failures:
                print(f"  - {line}")
            return 2
        print("Threshold check passed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
