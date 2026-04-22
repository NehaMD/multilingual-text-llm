import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Configuration variables
BATCH_SIZE = 32
SEQ_LEN = 128
VOCAB_SIZE = 8000
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4  # New: Number of Transformer blocks

def scaled_dot_product_attention(query, key, value, mask=None):
    """Computes Scaled Dot-Product Attention.

    Args:
        query (torch.Tensor): Query tensor of shape [Batch, Heads, SeqLen, HeadDim].
        key (torch.Tensor): Key tensor of shape [Batch, Heads, SeqLen, HeadDim].
        value (torch.Tensor): Value tensor of shape [Batch, Heads, SeqLen, HeadDim].
        mask (torch.Tensor, optional): Mask tensor of shape [Batch, Heads, SeqLen, SeqLen] 
            or broadcastable. Defaults to None.

    Returns:
        tuple: (context, attention_weights)
            context (torch.Tensor): Output tensor of shape [Batch, Heads, SeqLen, HeadDim].
            attention_weights (torch.Tensor): Attention scores of shape [Batch, Heads, SeqLen, SeqLen].
    """
    d_k = key.shape[-1]
    
    # Matrix multiplication of Queries and Keys to compute raw attention scores
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # Scaling the scores by the square root of the head dimension to stabilize gradients
    scaled_scores = scores / math.sqrt(d_k)
    
    if mask is not None:
        # Apply causal mask by filling future positions with a very large negative value
        # Mask is 0 for positions to be hidden, 1 for visible
        scaled_scores = scaled_scores.masked_fill(mask == 0, -1e9)
    
    # Applying Softmax to obtain probability distribution (attention weights)
    attention_weights = F.softmax(scaled_scores, dim=-1)
    
    # Computing the weighted sum of Values based on attention weights
    context = torch.matmul(attention_weights, value)
    
    return context, attention_weights

class TokenAndPositionEmbedding(nn.Module):
    """Combines Token Embeddings with Learned Position Embeddings.

    Args:
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimension of the embedding space.
        max_seq_len (int): Maximum sequence length supported by position embeddings.
    """
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        
    def forward(self, x):
        """Forward pass for Token and Position Embeddings.

        Args:
            x (torch.Tensor): Input tensor of token indices with shape [Batch, SeqLen].

        Returns:
            torch.Tensor: Combined embeddings with shape [Batch, SeqLen, EmbedDim].
        """
        seq_len = x.shape[1]
        tok_emb = self.token_emb(x)
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        pos_emb = self.pos_emb(positions)
        return tok_emb + pos_emb

class MultiHeadAttention(nn.Module):
    """Implements Multi-Head Causal Self-Attention.

    Args:
        embed_dim (int): Dimension of the embedding space.
        num_heads (int): Number of attention heads.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "EMBED_DIM must be divisible by NUM_HEADS"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Three linear layers for Q, K, V
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        
        # Final output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Register a lower-triangular causal mask buffer
        # This creates a [1, 1, SEQ_LEN, SEQ_LEN] mask to ensure broadcasting across batch and heads
        mask = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN))
        self.register_buffer("causal_mask", mask.view(1, 1, SEQ_LEN, SEQ_LEN))
        
    def forward(self, x, return_attention=False):
        """Forward pass for MultiHeadAttention.
        Args:
            x (torch.Tensor): Input tensor of shape [Batch, SeqLen, EmbedDim].
            return_attention (bool): If True, returns (output, attention_weights).
        Returns:
            torch.Tensor or tuple: Output tensor or (output, attention_weights).
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections to obtain Query, Key, and Value tensors
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        
        # Reshaping and transposing to split the embedding dimension into multiple heads
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention with causal mask
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        context, attention_weights = scaled_dot_product_attention(q, k, v, mask=mask)
        
        # Concatenating the heads back into a single tensor
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Final linear projection
        output = self.out_proj(context)

        if return_attention:
            return output, attention_weights
        return output

class FeedForward(nn.Module):
    """Implements the Position-wise Feed-Forward Network.

    Args:
        embed_dim (int): Dimension of the embedding space.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
    """
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        # Expansion factor is typically 4x in Transformers
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """Forward pass for the Feed-Forward Network.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch, SeqLen, EmbedDim].

        Returns:
            torch.Tensor: Output tensor after non-linear transformation with shape [Batch, SeqLen, EmbedDim].
        """
        return self.net(x)

class TransformerBlock(nn.Module):
    """Implements a single Transformer Encoder/Decoder Block.

    Args:
        embed_dim (int): Dimension of the embedding space.
        num_heads (int): Number of attention heads.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        # Layer Normalization layers
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        
        # Attention and FeedForward components
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim, dropout)
        
    def forward(self, x, return_attention=False):
        """Forward pass for the Transformer Block.
        Args:
            x (torch.Tensor): Input tensor of shape [Batch, SeqLen, EmbedDim].
            return_attention (bool): If True, also returns attention weights.
        """
        # 1. Pre-LayerNorm Attention with Residual Connection
        ln_x = self.ln1(x)
        if return_attention:
            attn_out, attn_weights = self.mha(ln_x, return_attention=True)
            x = x + attn_out
        else:
            x = x + self.mha(ln_x)
        
        # 2. Pre-LayerNorm Feed-Forward with Residual Connection
        x = x + self.ff(self.ln2(x))
        
        if return_attention:
            return x, attn_weights
        return x

class LanguageModel(nn.Module):
    """Implements a Decoder-only Transformer Language Model.

    Args:
        vocab_size (int): Size of the vocabulary.
        max_seq_len (int): Maximum sequence length.
        embed_dim (int): Dimension of the embedding space.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of Transformer blocks.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
    """
    def __init__(self, vocab_size, max_seq_len, embed_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        # Initial Embedding layer
        self.embedding = TokenAndPositionEmbedding(vocab_size, embed_dim, max_seq_len)
        
        # Stack of Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        # Final LayerNorm before prediction head
        self.ln_f = nn.LayerNorm(embed_dim)
        
        # Linear classification head (logits)
        self.head = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x, return_attention=False):
        """Forward pass for the Language Model.
        Args:
            x (torch.Tensor): Input tensor of token indices.
            return_attention (bool): If True, returns (logits, all_attentions).
        """
        all_attentions = []
        
        # 1. Input IDs -> Embeddings
        x = self.embedding(x)
        
        # 2. Pass through Transformer blocks
        for block in self.blocks:
            if return_attention:
                x, attn = block(x, return_attention=True)
                all_attentions.append(attn)
            else:
                x = block(x)
            
        # 3. Final normalization
        x = self.ln_f(x)
        
        # 4. Final classification head
        logits = self.head(x)
        
        if return_attention:
            return logits, all_attentions
        return logits

    @torch.no_grad()
    def get_embeddings(self, x):
        """Extracts latent representations from the final Transformer block.
        Args:
            x (torch.Tensor): Input token indices [Batch, SeqLen].
        Returns:
            torch.Tensor: Latent embeddings [Batch, SeqLen, EmbedDim].
        """
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return x

if __name__ == "__main__":
    # 1. Generate dummy input and target data
    # dummy_x: [Batch, SeqLen] - input tokens
    # target_y: [Batch, SeqLen] - the tokens we want the model to predict
    dummy_x = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))
    target_y = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))
    print(f"Input shape: {dummy_x.shape}, Target shape: {target_y.shape}")

    # 2. Instantiate the full LanguageModel
    model = LanguageModel(
        vocab_size=VOCAB_SIZE,
        max_seq_len=SEQ_LEN,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS
    )
    
    # 3. Setup Optimizer and Loss Function
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    print("\nStarting overfitting test (50 iterations)...")
    model.train()
    
    for i in range(50):
        optimizer.zero_grad()
        
        # Forward pass: [Batch, SeqLen, VocabSize]
        logits = model(dummy_x)
        
        # Reshape for CrossEntropyLoss:
        # Logits: [Batch * SeqLen, VocabSize]
        # Targets: [Batch * SeqLen]
        loss = criterion(logits.view(-1, VOCAB_SIZE), target_y.view(-1))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 10 == 0:
            print(f"Step {i+1}/50 | Loss: {loss.item():.4f}")

    print("\nOverfitting test complete.")
    if loss.item() < 1.0:
        print("Success! The model is learning (loss decreased significantly).")
    else:
        print("Warning: Loss did not decrease as expected. Check architecture/gradients.")
