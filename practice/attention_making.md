That’s a sharp observation! Here’s how attention works in generation despite examining the full sequence.

### 💡 **1. Attention in Transformers: Full Sequence Context**
In the standard Transformer architecture (as introduced in the original paper by Vaswani et al., 2017), the attention mechanism processes **the entire input sequence at once**. This is particularly true for **encoder-decoder architectures**:
- The **encoder** attends to the full input sequence.
- The **decoder** generates tokens by attending to both the **encoder's output** and the previously generated tokens.

However, when it comes to **generation tasks** (like text generation or autoregressive decoding), the model doesn’t process the entire sequence in a single step. Instead, it generates one token at a time in an **autoregressive** manner, while **only attending to the tokens generated so far**.

---

### 🔥 **2. Generation Uses Causal Masking**
During autoregressive generation:
- The decoder uses **causal masking** (also called look-ahead masking) to **prevent attending to future tokens**. 
- The attention matrix is masked so that token \( t \) can only attend to tokens \( \leq t \) but **not future tokens**.
- This ensures that the model generates tokens sequentially rather than all at once.

✅ **Causal Masking Example**  
For a sequence \( x = [x_1, x_2, x_3, x_4] \), at time step \( t = 2 \), the model can only attend to \( x_1 \) and \( x_2 \), but not \( x_3 \) or \( x_4 \). The attention mask would look like this for a 4-token sequence:

\[
\text{{Mask}} = 
\begin{bmatrix} 
1 & 0 & 0 & 0 \\ 
1 & 1 & 0 & 0 \\ 
1 & 1 & 1 & 0 \\ 
1 & 1 & 1 & 1 
\end{bmatrix}
\]

---

### 🚀 **3. How Attention Applies During Generation**
- **During inference**, the model generates one token at a time, appending it to the current sequence.
- The self-attention mechanism then recalculates attention scores over the **entire generated sequence so far**, but not beyond.
- This is why models like **GPT** (which are decoder-only) use causal masking to prevent cheating by peeking into future tokens.

---

### ⚙️ **4. Why Full-Sequence Attention is Still Used**
Even though generation is autoregressive, the attention mechanism still applies to the **entire sequence seen so far**. This is because:
- Each generated token benefits from the full context of all previous tokens.
- The model re-evaluates the entire self-attention pattern for the expanded sequence after every new token is appended.
- This allows it to maintain consistency, coherence, and continuity over the generated text.

---

### 🛠️ **5. Key Difference: Training vs. Inference**
- During **training**, transformers often process the entire sequence at once with **full attention** (no causal masking in the encoder, but masking in the decoder). This enables efficient parallelization.
- During **generation (inference)**, tokens are generated one by one using autoregressive decoding with causal masking, despite the model itself being trained with full-sequence attention.

---

### ✅ **TL;DR**
- During **generation**, transformers use **causal masking** to prevent attention to future tokens.
- The model **recalculates attention** over the full sequence generated so far after appending each new token.
- This makes attention applicable in generation, as the model considers the entire context of previously generated tokens for each new prediction.