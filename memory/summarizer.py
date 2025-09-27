import re
from transformers import pipeline

# Lazy-load once (cached after first use)
_summarizer = None

def summarize_text(text: str, style: str = "bullet") -> str:
    # For short texts, use simple formatting - no need for complex summarization
    if len(text.split()) < 10:
        if style == "bullet":
            return f"- {text}"
        return text

    # Use extractive summarization instead of abstractive (BART)
    # This prevents the model from adding its own content
    try:
        # Simple extractive approach - take the first meaningful sentence
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        meaningful_sentences = [s for s in sentences if len(s.split()) > 4]
        
        if not meaningful_sentences:
            if style == "bullet":
                return f"- {text[:100]}..." if len(text) > 100 else f"- {text}"
            return text[:100] + "..." if len(text) > 100 else text
        
        if style == "bullet":
            # For bullet style, use up to 2 key sentences
            bullets = [f"- {s}" for s in meaningful_sentences[:2]]
            return "\n".join(bullets)
        else:
            # For concise style, use the first key sentence
            return meaningful_sentences[0]
            
    except Exception as e:
        # Fallback
        if style == "bullet":
            return f"- {text[:100]}..." if len(text) > 100 else f"- {text}"
        return text[:100] + "..." if len(text) > 100 else text

if __name__ == "__main__":
    test_text = (
        "CognitiveGPT is an AI agent that mimics human working memory. "
        "It can forget, summarize, and retrieve old information just like a brain. "
        "The user wants to simulate ADHD and photographic memory too."
    )
    summary = summarize_text(test_text, style="bullet")
    print("📋 Summary:\n", summary)