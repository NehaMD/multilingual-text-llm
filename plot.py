import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sentencepiece as spm
from model import LanguageModel
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine
import os

# -----------------------------
# CONFIGURATION (Match your model.pth)
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
MODEL_PATH = "model.pth"
TOKENIZER_PATH = "tokenizer.model"

# Hyperparameters (MUST match training)
VOCAB_SIZE = 16000
BLOCK_SIZE = 128
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4

# -----------------------------
# LOAD MODEL & TOKENIZER
# -----------------------------
def load_assets():
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)
    
    model = LanguageModel(VOCAB_SIZE, BLOCK_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print(f"WARNING: {MODEL_PATH} not found. Using randomly initialized weights.")
    
    model.to(DEVICE)
    model.eval()
    return model, sp

# -----------------------------
# 1. ATTENTION HEATMAP
# -----------------------------
def plot_attention_heatmap(text, layer_idx=-1, head_idx=0, save_path="attention_heatmap.png"):
    model, sp = load_assets()
    
    tokens = sp.encode(text, add_bos=True)
    token_labels = [sp.decode([t]) for t in tokens]
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)
    
    _, attentions = model(x, return_attention=True)
    # attentions shape: [batch, heads, seq_len, seq_len]
    attn = attentions[layer_idx][0, head_idx].detach().cpu().numpy()
    
    # Crop to current text length
    seq_len = len(tokens)
    attn = attn[:seq_len, :seq_len]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn, xticklabels=token_labels, yticklabels=token_labels, cmap="viridis", annot=False)
    plt.title(f"Attention Heatmap (Layer {layer_idx % NUM_LAYERS}, Head {head_idx})")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved attention heatmap to {save_path}")
    plt.close()

# -----------------------------
# 2. EMBEDDING SIMILARITY MATRIX
# -----------------------------
def plot_similarity_matrix(word_list_1, word_list_2, save_path="similarity_matrix.png"):
    model, sp = load_assets()
    
    def get_vec(word):
        tokens = torch.tensor(sp.encode(word, add_bos=True), dtype=torch.long).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = model.get_embeddings(tokens)
            # Use the representation of the last token
            return emb[0, -1, :].cpu().numpy()

    vecs1 = [get_vec(w) for w in word_list_1]
    vecs2 = [get_vec(w) for w in word_list_2]
    
    sim_matrix = np.zeros((len(word_list_1), len(word_list_2)))
    for i, v1 in enumerate(vecs1):
        for j, v2 in enumerate(vecs2):
            sim_matrix[i, j] = 1 - cosine(v1, v2) # Cosine Similarity
            
    plt.figure(figsize=(12, 10))
    sns.heatmap(sim_matrix, xticklabels=word_list_2, yticklabels=word_list_1, cmap="coolwarm", annot=True, fmt=".2f")
    plt.title("Cross-Lingual Embedding Similarity")
    plt.xlabel("Language B")
    plt.ylabel("Language A")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved similarity matrix to {save_path}")
    plt.close()

# -----------------------------
# 3. T-SNE SEMANTIC CLUSTERS
# -----------------------------
def plot_tsne_clusters(word_groups, save_path="tsne_clusters.png"):
    model, sp = load_assets()
    
    words = []
    labels = []
    vectors = []
    
    for category, category_words in word_groups.items():
        for word in category_words:
            tokens = torch.tensor(sp.encode(word, add_bos=True), dtype=torch.long).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                emb = model.get_embeddings(tokens)
                vectors.append(emb[0, -1, :].cpu().numpy())
                words.append(word)
                labels.append(category)
                
    vectors = np.array(vectors)
    
    # Run T-SNE
    # Reduce perplexity if we have very few words
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(words)-1), init='pca', learning_rate='auto')
    reduced_vecs = tsne.fit_transform(vectors)
    
    plt.figure(figsize=(12, 8))
    unique_labels = list(set(labels))
    colors = plt.cm.get_cmap("tab10", len(unique_labels))
    
    for i, category in enumerate(unique_labels):
        indices = [j for j, l in enumerate(labels) if l == category]
        plt.scatter(reduced_vecs[indices, 0], reduced_vecs[indices, 1], label=category, s=100)
        
        for idx in indices:
            plt.annotate(words[idx], (reduced_vecs[idx, 0], reduced_vecs[idx, 1]), xytext=(5, 2), textcoords='offset points')
            
    plt.legend()
    plt.title("T-SNE Visualization of Multilingual Embeddings")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved T-SNE plot to {save_path}")
    plt.close()

# -----------------------------
# MAIN DEMO
# -----------------------------
if __name__ == "__main__":
    # 1. Attention Demo
    sample_text = "The quick brown fox jumps over the lazy dog. नमस्ते दुनिया।"
    plot_attention_heatmap(sample_text)
    
    # 2. Similarity Demo (Synonyms across languages)
    english_words = ["water", "sun", "food", "hello", "book"]
    hindi_words = ["पानी", "सूरज", "खाना", "नमस्ते", "किताब"]
    plot_similarity_matrix(english_words, hindi_words)
    
    # 3. T-SNE Demo (Categorical clusters)
    groups = {
        "Nature": ["tree", "पेड़", "leaf", "पत्ती", "river", "नदी"],
        "Animals": ["cat", "बिल्ली", "dog", "कुत्ता", "lion", "शेर"],
        "Numbers": ["one", "एक", "two", "दो", "three", "तीन"]
    }
    plot_tsne_clusters(groups)
    
    print("\nAll visualizations generated successfully!")
