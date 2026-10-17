"""Run every question in questions.json through the RAG pipeline and score it.

Run from the project root:
    python evals/run_evals.py                   # full run: retrieval + Claude (costs a few cents)
    python evals/run_evals.py --retrieval-only  # free: checks retrieval only, no API calls
    python evals/run_evals.py --top-k 10        # try a different setting

Three scores:
  retrieval hit rate  did the chunk holding the answer reach Claude? (answerable questions)
  answer accuracy     did Claude's answer contain the expected keywords? (answerable questions)
  refusal accuracy    did Claude say "I don't know" when the answer isn't in the docs?
"""
import argparse
import json
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import rag  # noqa: E402

QUESTIONS_FILE = HERE / "questions.json"
RESULTS_DIR = HERE / "results"
REFUSAL = "i don't know"
PRICE_IN, PRICE_OUT = 0.10, 0.50  # Claude Haiku 5.5, USD per million tokens (prompts under 100k)


def normalize(text):
    """Lower-case, turn ligatures like 'ﬁ' into 'fi', straighten quotes, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).replace("’", "'")
    return re.sub(r"\s+", " ", text).lower()


def answer_is_correct(answer, expected):
    """expected is a list of groups; every group needs at least one of its keywords in the answer."""
    a = normalize(answer)
    return all(any(normalize(k) in a for k in group) for group in expected)


def is_refusal(answer):
    return REFUSAL in normalize(answer)


def retrieval_hit(sources, evidence):
    """True if the passage holding the answer was among the retrieved chunks."""
    e = normalize(evidence)
    return any(e in normalize(s["text"]) for s in sources)


def mark(ok):
    return "-" if ok is None else ("✓" if ok else "✗")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=rag.TOP_K)
    parser.add_argument("--retrieval-only", action="store_true", help="skip Claude; score retrieval only (free)")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_FILE.read_text())
    index = rag.build_index()

    # Sanity check: an evidence phrase that appears in NO chunk means the question
    # itself needs fixing (typo, or the PDF text differs), not that retrieval failed.
    all_text = normalize(" ".join(c["text"] for c in index["chunks"]))
    for q in questions:
        if q["answerable"] and normalize(q["evidence"]) not in all_text:
            print(f"WARNING {q['id']}: evidence {q['evidence']!r} not found anywhere in the index")

    client = None if args.retrieval_only else rag.anthropic.Anthropic()
    rows = []
    print(f"\n{'id':10} {'R':1} {'A':1} {'tokens':>7}  answer")
    for q in questions:
        row = {"id": q["id"], "question": q["question"], "answerable": q["answerable"]}
        if args.retrieval_only:
            sources = rag.retrieve(index, q["question"], args.top_k)
        else:
            try:
                result = rag.answer(index, q["question"], args.top_k, client=client)
            except Exception as err:  # keep going if one call fails
                row["error"] = str(err)
                rows.append(row)
                print(f"{q['id']:10} ERROR {err}")
                continue
            sources = result["sources"]
            row.update(answer=result["answer"], input_tokens=result["input_tokens"],
                       output_tokens=result["output_tokens"], stop_reason=result["stop_reason"])

        row["sources"] = [{"source": s["source"], "page": s["page"], "score": round(s["score"], 3)} for s in sources]
        row["retrieval_hit"] = retrieval_hit(sources, q["evidence"]) if q["answerable"] else None

        if "answer" in row:
            if q["answerable"]:
                row["correct"] = (not is_refusal(row["answer"])) and answer_is_correct(row["answer"], q["expected"])
            else:
                row["correct"] = is_refusal(row["answer"])
        else:
            row["correct"] = None
        rows.append(row)

        tokens = row.get("input_tokens", 0) + row.get("output_tokens", 0)
        short = row.get("answer", "").replace("\n", " ")[:70]
        print(f"{q['id']:10} {mark(row['retrieval_hit'])} {mark(row['correct'])} {tokens:>7}  {short}")

    # ---------- Summary ----------
    answerable = [r for r in rows if r["answerable"] and "error" not in r]
    unanswerable = [r for r in rows if not r["answerable"] and "error" not in r]
    summary = {
        "retrieval_hit_rate": f"{sum(r['retrieval_hit'] for r in answerable)}/{len(answerable)}",
    }
    print("\nSUMMARY")
    print(f"  retrieval hit rate : {summary['retrieval_hit_rate']}")
    if not args.retrieval_only:
        in_tok = sum(r.get("input_tokens", 0) for r in rows)
        out_tok = sum(r.get("output_tokens", 0) for r in rows)
        cost = in_tok / 1e6 * PRICE_IN + out_tok / 1e6 * PRICE_OUT
        summary.update(
            answer_accuracy=f"{sum(r['correct'] for r in answerable)}/{len(answerable)}",
            refusal_accuracy=f"{sum(r['correct'] for r in unanswerable)}/{len(unanswerable)}",
            input_tokens=in_tok, output_tokens=out_tok, cost_usd=round(cost, 4),
        )
        print(f"  answer accuracy    : {summary['answer_accuracy']}")
        print(f"  refusal accuracy   : {summary['refusal_accuracy']}")
        print(f"  tokens             : {in_tok} in / {out_tok} out  (about ${cost:.4f})")
    errors = [r["id"] for r in rows if "error" in r]
    if errors:
        print(f"  errors             : {', '.join(errors)}")

    # ---------- Save, so every experiment leaves a record you can compare ----------
    config = {"chunk_size": rag.CHUNK_SIZE, "chunk_overlap": rag.CHUNK_OVERLAP, "top_k": args.top_k,
              "embed_model": rag.EMBED_MODEL,
              "claude_model": None if args.retrieval_only else rag.CLAUDE_MODEL}
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({"config": config, "summary": summary, "results": rows}, indent=2, ensure_ascii=False))
    print(f"\nsaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
