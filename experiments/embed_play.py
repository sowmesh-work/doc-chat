from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "The comparison is performed on a 5 million packet CAIDA trace.",
    "Bloom filters can produce false positives.",
    "Alexander crossed the river Granicus with his cavalry.",
]
question = "What dataset did the authors test on?"

sentence_vecs = model.encode(sentences)
question_vec = model.encode(question)

print(len(question_vec))
print(util.cos_sim(question_vec, sentence_vecs))