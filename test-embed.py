from google import genai

client = genai.Client(
    vertexai=True,
    project="project-ff7c2ef5-8d88-401a-b86",
    location="us-central1",
)

result = client.models.embed_content(model="text-embedding-004", contents=["select 1"])
print("embeddings returned:", len(result.embeddings))
print("vector length:", len(result.embeddings[0].values))
print("first 5 values:", result.embeddings[0].values[:5])c