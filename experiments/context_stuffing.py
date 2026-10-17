"""Experiment: send the whole Bloom paper to Claude in one prompt, with no retrieval.

This is the baseline to compare RAG against. Run from the project root:
    python experiments/context_stuffing.py
"""
import anthropic
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()
client = anthropic.Anthropic()

SYSTEM_PROMPT = """You answer questions about a research paper.

Rules:
1. Answer only from the research paper.
2. If the answer is not in the paper, reply exactly: I don't know
3. Use the paper's own wording for terms and definitions."""

reader = PdfReader("data/bloom.pdf")
page_text = []
for page in reader.pages:
    page_text.append(page.extract_text())
full_text = "\n\n".join(page_text)

question = "What dataset did the authors test on?"
prompt = f"""Here is the research paper:
<paper>{full_text}</paper>
Question: {question}"""

message = client.messages.create(
    model="claude-haiku-5-5",
    max_tokens=1000,
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": prompt}],
)

for block in message.content:
    if block.type == "text":
        print(block.text)
print(message.stop_reason)
print(message.usage)
