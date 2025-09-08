import re
from transformers import pipeline

# Lazy-load once (cached after first use)
_summarizer = None

def summarize_text(text: str, style: str = "bullet") -> str:
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    # Tweak lengths a bit for short inputs
    if style == "concise":
        max_len, min_len = 80, 20
    else:
        max_len, min_len = 128, 30

    out = _summarizer(
        text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False
    )[0]["summary_text"].strip()

    if style == "bullet":
        # Turn summary into up to 3 bullets
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', out) if s.strip()]
        bullets = [f"- {s}" for s in sents[:3]] or [f"- {out}"]
        return "\n".join(bullets)

    return out
