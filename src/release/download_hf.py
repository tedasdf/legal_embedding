from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "Sing0402/legal-embed-gte-inbatch",
    device="cpu",
    trust_remote_code=True,
)

embedding = model.encode(
    ["What constitutes a breach of contract?"],
    normalize_embeddings=True,
)

print(embedding.shape)
print("Norm:", (embedding[0] ** 2).sum() ** 0.5)