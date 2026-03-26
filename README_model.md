# The Beginner's Guide to our Transformer Language Model

This guide explains the inner workings of `model.py`. If you have no background in Deep Learning, don't worry—this document is designed to take you from "What is a tensor?" to understanding how a GPT-style model "thinks."

---

## Prologue: What is a "Transformer"?

Before 2017, AI processed text like a human reading a book: one word at a time, from left to right. This was slow and the AI often "forgot" the beginning of a sentence by the time it reached the end.

The **Transformer** changed everything. Instead of reading sequentially, it looks at **every word in a sentence simultaneously**. It uses a mechanism called "Attention" to decide which words are relevant to each other, regardless of how far apart they are.

---

## 1. Architecture Overview: The "Next-Word" Machine

Our model is a **Decoder-only Transformer**. You can think of it as a highly sophisticated "fill-in-the-blanks" machine. 

### Why "Decoder-only"?
In the world of AI, there are "Encoders" (which understand text) and "Decoders" (which generate text). Since our goal is to build a Large Language Model (LLM) that can write stories or answer questions, we use the Decoder-only architecture. It is **auto-regressive**, meaning it predicts the very next word based on all the words that came before it.

---

## 2. The Data Contract: Talking to the Model

Computers don't understand words like "Apple" or "Namaste." They only understand numbers.

### What is a Tensor?
Think of a **Tensor** as a multi-dimensional grid of numbers. 
- A 1D tensor is a list (Vector).
- A 2D tensor is a table (Matrix).
- A 3D tensor is a stack of tables (a "Cube" of data).

### The Variables
- **`BATCH_SIZE` (4):** We don't feed the model one sentence at a time; we feed it 4 sentences at once to make training faster.
- **`SEQ_LEN` (128):** Each sentence is exactly 128 words long. If a sentence is shorter, we pad it with empty space.
- **`VOCAB_SIZE` (5000):** Our model knows exactly 5,000 unique "tokens" (words or parts of words).
- **`EMBED_DIM` (256):** Every word is represented by a list of 256 numbers.

### The Input/Output Contract
- **Input (X):** A table of shape `[4, 128]`. It contains 4 sentences, each with 128 word-IDs.
- **Output (Logits):** A 3D cube of shape `[4, 128, 5000]`. For every word in every sentence, the model gives a "score" to all 5,000 possible next words.

---

## 3. Component Deep-Dive: Building the Brain

### A. Embeddings: The "Map of Meaning"
**Class:** `TokenAndPositionEmbedding`
If you plot words on a map, "King" and "Queen" would be very close to each other, while "Apple" would be far away. An **Embedding** is a coordinate system for meaning. 
- **The Problem:** A Transformer looks at all words at once, so it doesn't know which word comes first. 
- **The Solution:** We add **Position Embeddings**. Think of this as giving every word a "page number" so the model knows its place in the sequence.

### B. Scaled Dot-Product Attention: The "Filing Cabinet" Analogy
**Function:** `scaled_dot_product_attention`
This is the heart of the model. It uses three concepts:
1.  **Query (Q):** What I am looking for.
2.  **Key (K):** What I have to offer.
3.  **Value (V):** The actual information.

**Analogy:** Imagine you are in a library. Your **Query** is the topic you want to research. The **Keys** are the labels on the spines of the books. You compare your Query to all the Keys. When you find a match, you pull the **Value** (the information inside the book).

**The "Scaling" Part:** We divide the math by the square root of the dimension. Why? If the numbers get too big, the model's "brain" gets overwhelmed (gradients vanish), and it stops learning. Scaling keeps the signals manageable.

### C. Multi-Head Attention: Parallel Perspectives
**Class:** `MultiHeadAttention`
One "head" of attention might focus on **Grammar** (Who is the subject?). Another might focus on **Context** (Is this a happy or sad sentence?). By having 8 heads, the model can "think" about 8 different aspects of the language at the exact same time.

**The Causal Mask (The "No Peeking" Rule):** 
When we train the model to predict the next word, we can't let it see the answer! The Causal Mask is like putting blinders on a horse. It hides all future words so the model is forced to actually learn how to predict.

### D. Feed-Forward Networks: The "Fact-Checker"
**Class:** `FeedForward`
After the words have "talked" to each other via Attention, each word goes through a Feed-Forward network. Think of this as a private office where each word processes what it just learned. 
- We use **GELU**, a mathematical curve that decides which information is important enough to pass forward.

### E. Transformer Block: The Assembly Line
**Class:** `TransformerBlock`
This combines Attention and Feed-Forward into one unit. 
- **Residual Connections:** We always add the original input back to the result ($x + f(x)$). This is a "safety rope"—if the layer makes a mistake, the original information can still flow through.
- **Layer Normalization:** This keeps the numbers in a healthy range (not too big, not too small) so the model doesn't "explode" during training.

---

## 4. The "Lifecycle of a Word": A Forward Pass Trace

Let's follow a single word, **"Namaste"**, as it travels through the model:

1.  **Entry:** "Namaste" enters as a simple ID number (e.g., `402`).
2.  **Mapping:** It's turned into a list of 256 numbers and gets its "page number" added.
3.  **Attention:** "Namaste" looks at the other words in the sentence. It realizes the word "India" is nearby and strengthens its connection to it.
4.  **Refinement:** It goes through the Feed-Forward "office" to sharpen its meaning.
5.  **Repetition:** This happens 4 times (since we have 4 layers).
6.  **The Vote:** In the final layer, the 256 numbers are projected into 5,000 scores.
7.  **Exit:** The model looks at the scores and says: "The most likely next word after 'Namaste' is 'Dosto' (Friends)!"

---

## Glossary of Terms for Beginners

- **Logits:** Raw scores. High score = "The model is confident."
- **Softmax:** A way to turn raw scores into percentages (e.g., 0.85% probability).
- **Gradients:** The "signal" used during training to tell the model how to fix its mistakes.
- **Dropout:** Randomly turning off some parts of the brain during training so it doesn't become too reliant on specific words (prevents "overfitting").
- **AdamW:** The "Brain Surgeon" (the optimizer) that actually goes in and tweaks the 256-dimension numbers to make the model smarter.
