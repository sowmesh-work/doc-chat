"""Retrieval only: find the chunks of the Bloom paper closest to a question.

Run from the project root:
    python search.py
"""
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util

from chunking import chunk_text

PDF_PATH = "data/bloom.pdf"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5
EMBED_MODEL = "all-MiniLM-L6-v2"
DEBUG_KEYWORD = "VBF:" # prints where the chunk containing this word ranks; set to None to turn off

# 1. Chunk each page, keeping its page number
reader = PdfReader(PDF_PATH)
all_chunks = []
for page_num, page in enumerate(reader.pages, start=1):
    for piece in chunk_text(page.extract_text(), CHUNK_SIZE, CHUNK_OVERLAP):
        all_chunks.append({"page": page_num, "text": piece, "source": "bloom.pdf"})

# 2. Embed every chunk
model = SentenceTransformer(EMBED_MODEL)
texts = [c["text"] for c in all_chunks]
chunk_vecs = model.encode(texts)

# 3. Embed the question and score it against every chunk
question = "What does VBF stand for?"
question_vec = model.encode(question)
scores = util.cos_sim(question_vec, chunk_vecs)[0]
top = scores.topk(TOP_K)

# Diagnostic: where did the chunk we expect end up?
if DEBUG_KEYWORD:
    for i, c in enumerate(all_chunks):
        if DEBUG_KEYWORD in c["text"]:
            rank = int((scores > scores[i]).sum()) + 1
            print(f"{DEBUG_KEYWORD} chunk: p.{c['page']} score={scores[i]:.3f} rank={rank} of {len(all_chunks)}")

# 4. Show the top results
for score, idx in zip(top.values, top.indices):
    c = all_chunks[int(idx)]
    print(f"{score:.3f} | p.{c['page']} | {c['text'][:150]}")