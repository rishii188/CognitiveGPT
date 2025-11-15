# CognitiveGPT (Working Progress)

This is a project on exploring how to simulate human working memory limits in AI agents. The goal is to restrict how much context the model can hold, then use summarisation and retrieval to maintain performance which is similar to how people manage short-term and long-term memory. Note: this project is not complete.

## Current components
- `memory_buffer.py`: fixed-size buffer that summarises messages when they are evicted.  
- `summarizer.py`: simple extractive summariser for short or long texts.  
- `vector_store.py`: embedding-based retrieval using `sentence-transformers/all-MiniLM-L6-v2` and cosine similarity.  

## Planned features
- Profiles for different memory styles (e.g. ADHD-like, photographic).  
- Streamlit interface for visualising memory state.  
- Evaluation tasks to measure reasoning performance under memory constraints.  

## Installation
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

Minimal `requirements.txt`:
```
numpy
scikit-learn
sentence-transformers
transformers
```

## Usage
Example (from `memory_buffer.py`):
```python
from memory_buffer import WorkingMemory

wm = WorkingMemory(buffer_size=2)
wm.add("This is the first message.")
wm.add("This is the second message.")
wm.add("This is the third message (the first one gets summarised).")

print("Buffer:", wm.get_memory())
print("Summaries:", wm.get_summaries())
```

Example (from `vector_store.py`):
```python
from vector_store import VectorMemoryStore

store = VectorMemoryStore()
store.add("User wants to build an AI that mimics human memory.")
store.add("User mentioned they often forget what they were doing.")

results = store.query("The user has trouble remembering things.", k=2)
print(results)
```

## Status
At the moment, the system can:
- Keep track of a fixed number of messages.  
- Summarise evicted messages.  
- Store and retrieve summaries based on similarity.  

The next stage is integrating these pieces into a controller and testing different memory profiles.




