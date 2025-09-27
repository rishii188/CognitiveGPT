import streamlit as st
from memory.memory_buffer import WorkingMemory
from memory.vector_store import VectorMemoryStore
from memory.summarizer import summarize_text

# Initialize with buffer_size=2 as intended
if "wm" not in st.session_state:
    st.session_state.wm = WorkingMemory(buffer_size=2, summarizer=summarize_text, style="bullet")

if "vs" not in st.session_state:
    st.session_state.vs = VectorMemoryStore(similarity_threshold=0.5)  # Increased threshold

wm = st.session_state.wm
vs = st.session_state.vs

st.set_page_config(page_title="Working Memory Simulator", layout="wide")
st.title("Working Memory Simulator")

if "history" not in st.session_state:
    st.session_state.history = []

# Add a clear button
if st.sidebar.button("Clear Memory"):
    wm.clear()
    st.session_state.vs = VectorMemoryStore(similarity_threshold=0.5)
    st.session_state.history = []
    st.rerun()

# Chat input
user_input = st.chat_input("Please say something.")

if user_input and user_input.strip():
    cleaned_input = ' '.join(user_input.split())
    
    # First check for relevant memories before adding new input
    working_memories = wm.get_memory()
    relevant_memories = []
    
    # Look for memory-related content in current working memory
    for memory in working_memories:
        if "memory systems" in memory.lower():
            relevant_memories.append(memory)
    
    # If not found in working memory, check vector store
    if not relevant_memories:
        retrieved = vs.query("memory systems")
        relevant_memories = retrieved
    
    # Now add the new message to working memory
    wm.add(cleaned_input)
    
    # Add new summaries to vector store if they're meaningful
    summaries = wm.get_summaries()
    if len(summaries) > len(vs.texts):
        for i in range(len(vs.texts), len(summaries)):
            if len(summaries[i].split()) > 3:
                vs.add(summaries[i])

    # Generate response
    if relevant_memories:
        reply = f"I remember you mentioned:\n\n"
        reply += "\n".join([f"- {memory}" for memory in relevant_memories])
        reply += f"\n\nIn response to your question: {cleaned_input}"
    else:
        reply = f"I'm processing: {cleaned_input}"
        if summaries:
            reply += "\n\nI don't have directly related memories, but I'm keeping track of our conversation."

    st.session_state.history.append(("user", cleaned_input))
    st.session_state.history.append(("assistant", reply))

# Show chat history
for speaker, msg in st.session_state.history:
    with st.chat_message(speaker):
        st.markdown(msg)

# Sidebar: show memory
with st.sidebar:
    st.header("🧠 Working Memory")
    memory_list = wm.get_memory()
    for i, m in enumerate(memory_list, 1):
        st.markdown(f"**{i}.** {m}")
    
    if not memory_list:
        st.caption("Working memory is empty")

    st.markdown("---")
    st.header("📝 Summarized Memory")
    summaries_list = wm.get_summaries()
    for i, s in enumerate(summaries_list, 1):
        st.markdown(f"**Summary {i}:**")
        st.markdown(s)
    
    if not summaries_list:
        st.caption("No summaries yet")