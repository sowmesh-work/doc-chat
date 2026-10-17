def chunk_text(text, size, overlap):
    if overlap >= size:
        raise ValueError("overlap must be smaller than size, or the loop never moves forward")
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i = i + size - overlap
    return chunks
