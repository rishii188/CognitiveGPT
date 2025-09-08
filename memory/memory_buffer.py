from collections import deque
import re
from summarizer import summarize_text  # <-- local import (works when run by path)

# --- simple local fallback if you ever want zero deps ---
def local_summarize(text: str, style: str = "bullet") -> str:
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if style == "concise":
        return (sents[0] if sents else text)[:160]
    bullets = [f"- {s}" for s in sents[:3]] or [f"- {text[:160]}..."]
    return "\n".join(bullets)

class WorkingMemory:
    def __init__(self, buffer_size=5, summarizer=None, style="bullet"):
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)
        self.summaries = []
        self.summarizer = summarizer or summarize_text  # default = HF summarizer
        self.summary_style = style

    def add(self, message: str):
        if len(self.memory) == self.buffer_size:
            evicted = self.memory[0]
            summary = self.summarizer(evicted, style=self.summary_style)
            self.summaries.append(summary)
        self.memory.append(message)

    def get_memory(self):
        return list(self.memory)

    def get_summaries(self):
        return self.summaries

if __name__ == "__main__":
    wm = WorkingMemory(buffer_size=3)  # uses HF summarizer by default
    wm.add("I am working on a new project.")
    wm.add("It's called CognitiveGPT.")
    wm.add("It simulates working memory.")
    wm.add("I want it to have ADHD mode.")

    print("Working Memory:", wm.get_memory())
    print("Summaries:")
    for s in wm.get_summaries():
        print(s)
