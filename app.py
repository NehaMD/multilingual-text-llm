import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sentencepiece as spm
from model import LanguageModel
import time

# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================
st.set_page_config(
    page_title="🌍 Multilingual LLM Visualizer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 18px;
        padding: 15px 30px;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .step-container {
        background-color: #f0f4ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    h2 {
        color: #667eea;
        border-bottom: 2px solid #667eea;
        padding-bottom: 10px;
    }
    .token-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        margin: 3px;
        font-weight: 600;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DEVICE & CONFIGURATION
# ============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Load tokenizer first to get actual vocab size
sp_temp = spm.SentencePieceProcessor()
try:
    sp_temp.load("tokenizer.model")
    actual_vocab_size = sp_temp.vocab_size()
except:
    actual_vocab_size = 16000

CONFIG = {
    "block_size": 256,
    "vocab_size": actual_vocab_size,
    "embed_dim": 512,
    "num_heads": 8,
    "num_layers": 8,
    "device": device
}

# ============================================================================
# CACHE LOADING
# ============================================================================
@st.cache_resource
def load_tokenizer():
    sp = spm.SentencePieceProcessor()
    sp.load("tokenizer.model")
    return sp

@st.cache_resource
def load_model_cached():
    model = LanguageModel(
        CONFIG["vocab_size"],
        CONFIG["block_size"],
        CONFIG["embed_dim"],
        CONFIG["num_heads"],
        CONFIG["num_layers"]
    )
    try:
        model.load_state_dict(torch.load("model.pth", map_location=device))
    except:
        st.error("⚠️ model.pth not found. Please train the model first.")
        return None
    model = model.to(device)
    model.eval()
    return model

# ============================================================================
# CORE FUNCTIONS
# ============================================================================
def encode(text, tokenizer):
    """Tokenize text to token IDs"""
    return tokenizer.encode(text, add_bos=True)

def decode(tokens, tokenizer):
    """Decode token IDs back to text"""
    return tokenizer.decode(tokens)

def get_attention_weights(model, tokens, layer_idx=0, head_idx=0):
    """Extract attention weights from a specific layer and head"""
    with torch.no_grad():
        tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
        
        # Forward pass through model
        x = model.token_pos_emb(tokens_tensor)
        
        for i, block in enumerate(model.transformer_blocks):
            if i == layer_idx:
                # Get attention from this layer
                attn_output, attn_weights = block.attn(x)
                # attn_weights shape: [batch, heads, seq_len, seq_len]
                return attn_weights[0, head_idx].cpu().numpy()
            x = block(x)
    return None

def generate_with_tracking(model, tokenizer, prompt, max_tokens=50, temperature=1.0, top_k=40):
    """Generate tokens with step tracking"""
    tokens = encode(prompt, tokenizer)
    original_token_count = len(tokens)
    generated_tokens = []
    logits_history = []
    
    with torch.no_grad():
        for step in range(max_tokens):
            tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
            tokens_cond = tokens_tensor[:, -CONFIG["block_size"]:]
            
            logits = model(tokens_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-K sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            generated_tokens.append(next_token)
            tokens.append(next_token)
            
            # Store top-5 probabilities for visualization
            top_probs, top_indices = torch.topk(probs, 5)
            logits_history.append({
                'step': step,
                'token': next_token,
                'top_tokens': top_indices[0].cpu().numpy(),
                'top_probs': top_probs[0].cpu().numpy()
            })
    
    return tokens, generated_tokens, logits_history

# ============================================================================
# HEADER
# ============================================================================
st.markdown("# 🌍 Multilingual LLM Visualization Tool")
st.markdown("### Visualizing Transformer Architecture, Tokenization & Text Generation")
st.divider()

# ============================================================================
# MAIN INTERFACE - TABS
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Text Generator", 
    "🔤 Tokenization", 
    "🧠 Model Architecture",
    "👁️ Attention Heatmaps",
    "📊 Generation Analytics"
])

# ============================================================================
# TAB 1: TEXT GENERATOR
# ============================================================================
with tab1:
    st.header("🎯 Interactive Text Generator")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        prompt = st.text_area(
            "Enter your prompt (English or Hindi):",
            value="The future of artificial intelligence",
            height=100,
            key="prompt_input"
        )
    
    with col2:
        st.markdown("### ⚙️ Parameters")
        max_tokens = st.slider("Max Tokens", 10, 200, 50, help="Maximum length of generated text")
        temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1, help="Controls randomness (0.1=focused, 2.0=creative)")
        top_k = st.slider("Top-K Sampling", 5, 100, 40, help="Only sample from top K tokens")
    
    if st.button("🚀 Generate Text", use_container_width=True):
        tokenizer = load_tokenizer()
        model = load_model_cached()
        
        if model is None:
            st.stop()
        
        with st.spinner("⏳ Generating... This may take a moment"):
            start_time = time.time()
            tokens, gen_tokens, logits_info = generate_with_tracking(
                model, tokenizer, prompt, max_tokens, temperature, top_k
            )
            elapsed_time = time.time() - start_time
        
        # Display results
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Tokens Generated", len(gen_tokens), help="Number of new tokens created")
        with col2:
            st.metric("Generation Time", f"{elapsed_time:.2f}s", help="Time taken to generate")
        with col3:
            st.metric("Tokens/Sec", f"{len(gen_tokens)/elapsed_time:.2f}", help="Generation speed")
        
        st.markdown("---")
        
        # Display full output
        st.subheader("📝 Generated Output")
        full_text = decode(tokens, tokenizer)
        st.info(full_text, icon="✍️")
        
        # Display token breakdown
        st.subheader("🔍 Token Sequence")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Prompt Tokens:**")
            prompt_tokens = encode(prompt, tokenizer)
            for i, tok in enumerate(prompt_tokens[:20]):  # Show first 20
                token_text = tokenizer.decode([tok])
                st.caption(f"`{i}`: {token_text}")
        
        with col2:
            st.markdown("**Generated Tokens:**")
            for i, tok in enumerate(gen_tokens[:20]):  # Show first 20
                token_text = tokenizer.decode([tok])
                st.caption(f"`{i}`: {token_text}")

# ============================================================================
# TAB 2: TOKENIZATION VISUALIZATION
# ============================================================================
with tab2:
    st.header("🔤 Tokenization Process")
    
    text_input = st.text_input(
        "Enter text to tokenize:",
        value="Artificial Intelligence",
        key="tokenize_input"
    )
    
    if text_input:
        tokenizer = load_tokenizer()
        tokens = encode(text_input, tokenizer)
        
        st.markdown("---")
        
        # Show tokenization breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Tokenization Details")
            st.metric("Total Tokens", len(tokens), help="Including BOS token")
            st.metric("Vocabulary Size", CONFIG["vocab_size"], help="Total unique tokens")
            st.metric("Unique Tokens", len(set(tokens)), help="Unique tokens in this text")
        
        with col2:
            st.subheader("🎯 Token Visualization")
            df_tokens = []
            for i, token_id in enumerate(tokens):
                token_text = tokenizer.decode([token_id])
                df_tokens.append({"Position": i, "Token ID": token_id, "Text": token_text})
            
            st.dataframe(df_tokens, use_container_width=True, hide_index=True)
        
        # Character-level breakdown with colors
        st.subheader("🏗️ Token Structure")
        token_html = ""
        colors = ["#667eea", "#764ba2", "#f093fb", "#4facfe", "#00f2fe"]
        
        for i, token_id in enumerate(tokens):
            token_text = tokenizer.decode([token_id])
            color = colors[i % len(colors)]
            token_html += f'<span class="token-badge">{token_text}</span>'
        
        st.markdown(token_html, unsafe_allow_html=True)

# ============================================================================
# TAB 3: MODEL ARCHITECTURE
# ============================================================================
with tab3:
    st.header("🧠 Model Architecture Overview")
    
    # Architecture diagram
    st.subheader("📐 Architecture Layers")
    
    arch_data = {
        "Layer": [
            "Input Tokens",
            "Token Embedding",
            "Position Embedding",
            "Combined Embedding",
            "Transformer Block 1",
            "Transformer Block 2-8",
            "Final LayerNorm",
            "Output Logits"
        ],
        "Dimension": [
            "Variable",
            "512",
            "512",
            "512",
            "512",
            "512",
            "512",
            "16,000"
        ],
        "Parameters": [
            "0",
            "8,192,000",
            "65,536",
            "0",
            "3,000,000",
            "21,000,000",
            "1,024",
            "8,192,000"
        ]
    }
    
    # Display architecture table
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Parameters", "~21.7M", help="Total trainable parameters")
    with col2:
        st.metric("Vocabulary Size", "16,000", help="Custom BPE tokens")
    with col3:
        st.metric("Context Window", "256 tokens", help="Max sequence length")
    
    st.markdown("---")
    
    # Detailed component breakdown
    st.subheader("🔧 Component Details")
    
    component_col1, component_col2 = st.columns(2)
    
    with component_col1:
        st.markdown("### Multi-Head Attention")
        st.write(f"""
        - **Number of Heads**: {CONFIG['num_heads']}
        - **Head Dimension**: {CONFIG['embed_dim'] // CONFIG['num_heads']}
        - **Attention Type**: Causal (masked)
        - **Computation**: Q·K^T / √d_k + Mask
        """)
    
    with component_col2:
        st.markdown("### Feed Forward Network")
        st.write(f"""
        - **Input Dimension**: {CONFIG['embed_dim']}
        - **Hidden Dimension**: {CONFIG['embed_dim'] * 4}
        - **Output Dimension**: {CONFIG['embed_dim']}
        - **Activation**: GELU
        """)
    
    st.markdown("---")
    
    # Display architecture flow
    st.subheader("📊 Data Flow Visualization")
    st.markdown("""
    ```
    Input Text
        ↓
    [Tokenization] → Token IDs
        ↓
    [Token Embedding] + [Position Embedding]
        ↓
    [LayerNorm]
        ↓
    ┌─────────────────────────────────┐
    │   Transformer Block (×8)        │
    │  ┌─────────────────────────────┐│
    │  │ Multi-Head Attention        ││
    │  │  • Q, K, V Projections      ││
    │  │  • Scaled Dot-Product       ││
    │  │  • 8 Parallel Attention Heads││
    │  └──────────────┬──────────────┘│
    │                 ↓                │
    │           [Residual]             │
    │                 ↓                │
    │           [LayerNorm]            │
    │                 ↓                │
    │  ┌──────────────────────────────┐│
    │  │ Feed Forward Network         ││
    │  │  • Linear (→2048)            ││
    │  │  • GELU Activation           ││
    │  │  • Linear (→512)             ││
    │  └──────────────┬───────────────┘│
    │                 ↓                │
    │           [Residual]             │
    │                 ↓                │
    │           [LayerNorm]            │
    └──────────────┬──────────────────┘
                   ↓  (×8 times)
        [Final LayerNorm]
                   ↓
        [Linear Projection]
                   ↓
        [Softmax Distribution]
                   ↓
        [Next Token Prediction]
    ```
    """)

# ============================================================================
# TAB 4: ATTENTION HEATMAPS
# ============================================================================
with tab4:
    st.header("👁️ Attention Mechanism Visualization")
    
    attention_prompt = st.text_input(
        "Enter text to visualize attention weights:",
        value="Hello world",
        key="attention_input"
    )
    
    if attention_prompt:
        tokenizer = load_tokenizer()
        model = load_model_cached()
        
        if model is None:
            st.stop()
        
        tokens = encode(attention_prompt, tokenizer)
        
        # Select layer and head
        col1, col2 = st.columns(2)
        with col1:
            layer_idx = st.slider("Select Transformer Layer", 0, CONFIG["num_layers"]-1, 0)
        with col2:
            head_idx = st.slider("Select Attention Head", 0, CONFIG["num_heads"]-1, 0)
        
        # Get attention weights
        with st.spinner("Computing attention weights..."):
            attn_matrix = get_attention_weights(model, tokens, layer_idx, head_idx)
        
        if attn_matrix is not None:
            # Decode tokens for labels
            token_labels = [tokenizer.decode([t]) for t in tokens]
            
            # Create heatmap
            fig = ff.create_annotated_heatmap(
                z=attn_matrix,
                x=token_labels,
                y=token_labels,
                colorscale="Viridis",
                showscale=True,
                reversescale=False,
            )
            
            fig.update_layout(
                title=f"Attention Weights - Layer {layer_idx+1}, Head {head_idx+1}",
                xaxis_title="Key (Token Attended To)",
                yaxis_title="Query (Attending Token)",
                height=600,
                width=800,
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            st.subheader("📈 Attention Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Max Attention", f"{attn_matrix.max():.4f}")
            with col2:
                st.metric("Min Attention", f"{attn_matrix.min():.4f}")
            with col3:
                st.metric("Mean Attention", f"{attn_matrix.mean():.4f}")
            with col4:
                st.metric("Std Dev", f"{attn_matrix.std():.4f}")

# ============================================================================
# TAB 5: GENERATION ANALYTICS
# ============================================================================
with tab5:
    st.header("📊 Generation Analytics")
    
    st.subheader("🔍 Token Prediction Analysis")
    
    analytics_prompt = st.text_input(
        "Enter prompt for token analysis:",
        value="The future of",
        key="analytics_input"
    )
    
    num_steps = st.slider("Number of generation steps to analyze", 5, 50, 10)
    
    if st.button("Analyze Generation", use_container_width=True):
        tokenizer = load_tokenizer()
        model = load_model_cached()
        
        if model is None:
            st.stop()
        
        with st.spinner("Analyzing token predictions..."):
            tokens, gen_tokens, logits_info = generate_with_tracking(
                model, tokenizer, analytics_prompt, num_steps, temperature=0.8
            )
        
        # Create visualization of top-5 token probabilities over time
        steps = []
        probs_data = {i: [] for i in range(5)}
        token_labels = []
        
        for info in logits_info[:num_steps]:
            steps.append(info['step'])
            token_text = tokenizer.decode([info['token']])
            token_labels.append(f"Step {info['step']}: {token_text}")
            
            for rank, (prob, token_id) in enumerate(zip(info['top_probs'], info['top_tokens'])):
                probs_data[rank].append(float(prob))
        
        # Create line plot
        fig = go.Figure()
        
        colors = ["#667eea", "#764ba2", "#f093fb", "#4facfe", "#00f2fe"]
        for rank in range(5):
            fig.add_trace(go.Scatter(
                x=steps,
                y=probs_data[rank],
                mode='lines+markers',
                name=f'Top-{rank+1}',
                line=dict(color=colors[rank], width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title="Token Probability Distribution Over Generation Steps",
            xaxis_title="Generation Step",
            yaxis_title="Probability",
            hovermode='x unified',
            height=500,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Token selection summary
        st.subheader("🎯 Generated Token Sequence")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            generated_text = ""
            for token_id in gen_tokens[:num_steps]:
                token_text = tokenizer.decode([token_id])
                generated_text += token_text
            st.info(f"**Generated:** {generated_text}")
        
        with col2:
            st.metric("Tokens Analyzed", len(logits_info))

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: #667eea; font-weight: bold;'>
    🚀 Multilingual Transformer LLM Visualizer | Decoder-Only Architecture | 8 Layers | 21.7M Parameters
</div>
""", unsafe_allow_html=True)
