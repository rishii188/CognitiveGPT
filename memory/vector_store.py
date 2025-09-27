import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Initialize model once
_model = None

def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def get_embedding(text: str) -> np.ndarray:
    model = get_embedding_model()
    # Clean text before embedding
    clean_text = re.sub(r'^-\s*', '', text)  # Remove bullet points
    clean_text = clean_text.strip()
    embedding = model.encode(clean_text)
    return np.array(embedding, dtype=np.float32)

class VectorMemoryStore:
    def __init__(self, similarity_threshold=0.6):  # Increased threshold
        self.embeddings = []  # Store embeddings
        self.texts = []  # Store original summaries
        self.similarity_threshold = similarity_threshold

    def add(self, text):
        print(f"[Vector Store] Adding summary:\n{text}\n")
        vector = get_embedding(text)
        self.embeddings.append(vector)
        self.texts.append(text)

    def query(self, text, k=3):
        if len(self.texts) == 0:
            print("[Vector Store] No summaries stored yet.")
            return []

        print(f"\n🔎 [Vector Store] Query: {text}")
        
        query_vec = get_embedding(text)
        
        # Calculate cosine similarities
        similarities = cosine_similarity([query_vec], self.embeddings)[0]
        
        # Get indices of top k matches
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity >= self.similarity_threshold:
                print(f"✅ Match (similarity={similarity:.2f}): {self.texts[idx]}")
                results.append(self.texts[idx])
            else:
                print(f"❌ Below threshold (similarity={similarity:.2f}): {self.texts[idx]}")
            
        if not results:
            print("No meaningful matches.")
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