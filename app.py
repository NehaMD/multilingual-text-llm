import os
import torch
import torch.nn.functional as F
import sentencepiece as spm
import streamlit as st
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from model import LanguageModel

st.set_page_config(
    page_title="Multilingual LLM Visualizer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
body { background: #f8fbff; }
h1 { color: #3b4cc0; }
.token-badge { display:inline-block; margin:2px; padding:6px 12px; background:linear-gradient(135deg,#667eea,#764ba2); color:white; border-radius:999px; font-size:12px; }
</style>
""", unsafe_allow_html=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
MODEL_FILE = "model.pth"
TOKENIZER_FILE = "tokenizer.model"

@st.cache_resource
def load_tokenizer():
    if not os.path.exists(TOKENIZER_FILE):
        return None
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_FILE)
    return sp

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        return None
    tokenizer = load_tokenizer()
    vocab_size = tokenizer.vocab_size() if tokenizer is not None else 16000
    # Match the checkpoint parameters: vocab_size=16000, seq_len=128, embed_dim=128, heads=8, layers=4
    model = LanguageModel(vocab_size, 128, 128, 8, 4)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def encode(text, tokenizer):
    return tokenizer.encode(text, add_bos=True)

def decode(token_ids, tokenizer):
    return tokenizer.decode(token_ids)

def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, top_k=40):
    tokens = torch.tensor(encode(prompt, tokenizer), dtype=torch.long).unsqueeze(0).to(DEVICE)
    generated = []
    for _ in range(max_new_tokens):
        tokens_cond = tokens[:, -128:]  # Match model's max_seq_len
        logits = model(tokens_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat((tokens, next_token), dim=1)
        generated.append(next_token.item())
    return tokens[0].tolist(), generated

st.title("🌍 Multilingual LLM Visualization")
st.write("A minimal dashboard for generation, tokenization, and attention analysis.")

tokenizer = load_tokenizer()
model = load_model()

if tokenizer is None:
    st.warning("tokenizer.model is missing. Run `python tokenize_data.py` to create it from your dataset.")
if model is None:
    st.warning("model.pth is missing or cannot be loaded. Place your trained model file in the repo root.")

tab1, tab2 = st.tabs(["Text Generation", "Tokenization"])

with tab1:
    st.header("Text Generation")
    prompt = st.text_area("Prompt", value="The future of multilingual AI is", height=120)
    max_tokens = st.slider("Max tokens", 10, 100, 40)
    temperature = st.slider("Temperature", 0.1, 2.0, 0.8)
    top_k = st.slider("Top-K", 5, 100, 40)
    if st.button("Generate"):
        if model is None or tokenizer is None:
            st.error("Cannot generate because tokenizer.model or model.pth is missing.")
        else:
            with st.spinner("Generating..."):
                full_tokens, generated_ids = generate(model, tokenizer, prompt, max_tokens, temperature, top_k)
            st.subheader("Generated text")
            st.write(decode(full_tokens, tokenizer))
            st.subheader("Token sequence")
            token_texts = [tokenizer.decode([tok]) for tok in full_tokens]
            st.write(" ".join([f"`{i}:{t}`" for i, t in enumerate(token_texts)]))

with tab2:
    st.header("Tokenization")
    sample = st.text_input("Text to tokenize", value="Hello world")
    if st.button("Tokenize"):
        if tokenizer is None:
            st.error("tokenizer.model not found. Create it with `python tokenize_data.py`.")
        else:
            token_ids = encode(sample, tokenizer)
            st.write("**Token IDs:**", token_ids)
            st.write("**Decoded pieces:**")
            for idx, tok in enumerate(token_ids):
                st.markdown(f"- `{idx}` → `{tokenizer.decode([tok])}`")
            st.markdown("**Visual tokens**")
            row = "".join([f'<span class="token-badge">{tokenizer.decode([tok])}</span>' for tok in token_ids])
            st.markdown(row, unsafe_allow_html=True)
