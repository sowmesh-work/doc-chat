"""RAG over every PDF in data/: retrieve the most relevant chunks, then have
Claude answer from only those chunks, citing them.

Use it two ways:
    python rag.py "What dataset did the authors test on?"   # ask one question
    from rag import build_index, retrieve, answer            # from other scripts (evals)
"""
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util

from chunking import chunk_text

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# ---------- Settings: the knobs your experiments will change ----------
DATA_DIR = ROOT / "data"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5
EMBED_MODEL = "all-mpnet-base-v2"
CLAUDE_MODEL = "claude-haiku-5-5"

SYSTEM_PROMPT = """You answer questions using only the numbered sources provided.

Rules:
1. Answer only from the sources. Do not use outside knowledge.
2. After each claim, cite the source number(s) it came from, e.g. [2] or [1][3].
3. If the sources do not contain the answer, reply exactly: I don't know
4. Use the sources' own wording for terms and definitions."""


def build_index(data_dir=DATA_DIR, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Indexing: read every PDF, chunk each page, and embed every chunk.

    This is the slow, do-it-once step. Returns everything retrieval needs.
    """
    chunks = []
    for path in sorted(Path(data_dir).glob("*.pdf")):
        reader = PdfReader(path)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for piece in chunk_text(text, chunk_size, chunk_overlap):
                chunks.append({"source": path.name, "page": page_num, "text": piece})
        print(f"indexed {path.name}: {len(reader.pages)} pages", file=sys.stderr)

    model = SentenceTransformer(EMBED_MODEL)
    vectors = model.encode([c["text"] for c in chunks], show_progress_bar=True)
    print(f"{len(chunks)} chunks embedded", file=sys.stderr)
    return {"chunks": chunks, "vectors": vectors, "model": model}


def retrieve(index, question, top_k=TOP_K):
    """Query time, part 1: return the top_k chunks closest in meaning to the question."""
    question_vec = index["model"].encode(question)
    scores = util.cos_sim(question_vec, index["vectors"])[0]
    top = scores.topk(min(top_k, len(index["chunks"])))
    return [
        {**index["chunks"][int(i)], "score": float(s)}
        for s, i in zip(top.values, top.indices)
    ]


def build_prompt(question, retrieved):
    """Number each retrieved chunk as a source Claude can cite, then add the question."""
    sources = ""
    for n, c in enumerate(retrieved, start=1):
        sources += f"[{n}] ({c['source']}, page {c['page']})\n{c['text']}\n\n"
    return f"<sources>\n{sources}</sources>\n\nQuestion: {question}"


def answer(index, question, top_k=TOP_K, client=None):
    """Query time, part 2: retrieve, ask Claude, and return everything you'd want to inspect."""
    client = client or anthropic.Anthropic()
    retrieved = retrieve(index, question, top_k)
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(question, retrieved)}],
    )
    text = "".join(b.text for b in message.content if b.type == "text").strip()
    return {
        "question": question,
        "answer": text,
        "sources": retrieved,
        "stop_reason": message.stop_reason,
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
    }


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What dataset did the authors test on?"
    index = build_index()
    result = answer(index, question)

    print("\nANSWER")
    print(result["answer"])
    print("\nSOURCES")
    for n, s in enumerate(result["sources"], start=1):
        print(f"[{n}] {s['source']} p.{s['page']}  score={s['score']:.3f}")
    print(f"\nstop_reason={result['stop_reason']}")
    print(f"input_tokens={result['input_tokens']}  output_tokens={result['output_tokens']}")
