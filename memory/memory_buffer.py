from collections import deque
import re

def local_summarize(text: str, style: str = "bullet") -> str:
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    
    if style == "concise":
        return (sentences[0] if sentences else text)[:120]
    
    meaningful_sentences = [s for s in sentences if len(s.split()) > 3]
    bullets = [f"- {s}" for s in meaningful_sentences[:2]] or [f"- {text[:120]}..."]
    return "\n".join(bullets)

class WorkingMemory:
    def __init__(self, buffer_size=2, summarizer=None, style="bullet"):
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)
        self.summaries = []
        self.summarizer = summarizer or local_summarize
        self.summary_style = style

    def add(self, message: str):
        cleaned_message = ' '.join(message.split())
        
        # Check if adding will cause an eviction
        will_evict = len(self.memory) == self.buffer_size
        
        if will_evict:
            # Summarize the oldest message before it gets evicted
            oldest_message = self.memory[0] if self.memory else ""
            if len(oldest_message.split()) > 3:
                summary = self.summarizer(oldest_message, style=self.summary_style)
                self.summaries.append(summary)

        # Add the new message (this may automatically evict the oldest if deque is at maxlen)
        self.memory.append(cleaned_message)

    def get_memory(self):
        return list(self.memory)

    def get_summaries(self):
        return self.summaries
    
    def clear(self):
        self.memory.clear()
        self.summaries.clear()