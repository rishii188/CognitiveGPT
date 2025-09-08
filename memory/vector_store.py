import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

EMBED_MODEL = "text-embedding-3-small"  # or "text-embedding-ada-002"

def get_embedding(text: str) -> np.ndarray:
    embedding = model.encode(text)
    return np.array(embedding, dtype=np.float32)


class VectorMemoryStore:
    def __init__(self):
        self.index = faiss.IndexFlatL2(384)
        self.texts = [] # To store original summaries

    def add(self, text):
        vector = get_embedding(text)
        self.index.add(np.array([vector]))
        self.texts.append(text)

    def query(self, text, k=3):
        vector = get_embedding(text) # query vector
        distances, indices = self.index.search(np.array([vector]), k) # search the FAISS index and return 2D arrays distances (how far each match is) and indices (which items matched)

        results = []
        for i in indices[0]: # loop through first (and only) row of neighbours
            if i < len(self.texts):
                results.append(self.texts[i]) # grab text stored at i and store to results
        return results

if __name__ == "__main__":
    store = VectorMemoryStore()

    store.add("User wants to build an AI that mimics human memory.")
    store.add("User is struggling to stay focused on long tasks.")
    store.add("User mentioned they often forget what they were doing.")
    store.add("The AI uses GPT to summarize and recall past messages.")

    query = "The user has trouble remembering things."
    results = store.query(query, k=2)

    print("Query:", query)
    print("Top Matches:")
    for r in results:
        print("-", r)