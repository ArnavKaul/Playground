1.  **Input (The Human Stage)**
    * The user inputs a raw textual prompt into the system to initiate inference.
    * *Example:* `"The cat"`
2.  **Translation In (The Tokenizer: Encode)**
    * Because neural networks are purely mathematical machines that cannot natively process text, the Tokenizer's encoder splits the input into tokens and maps each token to its unique numerical index within a static lookup directory (`str_to_int`).
    * *Result:* `[14, 3, 8]` *(where "The" $\rightarrow$ 14, "cat" $\rightarrow$ 3, space $\rightarrow$ 8)*
3.  **The Brain (The Transformer Model Inference)**
    * The raw token IDs are fed directly into the neural network. The Transformer executes heavy tensor operations (matrix multiplications, multi-head attention passes, logit generation) to compute a probability distribution over the vocabulary. It predicts the single most statistically probable token ID to follow the input sequence.
    * *Result:* Model outputs a single brand-new number: `22`
4.  **Translation Out (The Tokenizer: Decode)**
    * The user cannot intuitively interpret what token ID `22` represents. The system passes the predicted ID back to the Tokenizer, which uses its reversed mapping directory (`int_to_str`) to convert the number back to human language and stitch the text back together.
    * *Result:* `22` maps back to `"sat"`. The user sees the final appended completion.

---

## 3. Byte-Pair Encoding (BPE) Deep Dive

### Architectural Goal
The fundamental objective of the BPE algorithm is to optimize token efficiency by systematically **replacing two adjacent, smaller blocks with a single, longer unified block**. This achieves radical text compression and guarantees that common character sequences cost fewer tokens during model processing.

### Scale Analogy
* **Small Scale (Local Practice):** When building a micro-transformer locally, a single text file (such as a *Taylor Swift Wikipedia page*) acts as the model's entire universe. The tokenizer scans *only* this corpus to construct its closed-world vocabulary, and the model's "knowledge" is strictly bounded by it.
* **Large Scale (Production LLMs):** For massive models like GPT-4, the "test file" is scaled up to the *entire internet corpus* scraped by OpenAI (billions of web pages, textbooks, and code repositories). The exact same BPE mechanics apply, but the vocabulary scales to roughly 50,000 to 100,000+ optimized subword fragments.

### Core Algorithmic Mechanics (Interview Blueprint)
1.  **Initialization:** The training corpus is initially broken down completely into individual characters. This serves as the base vocabulary.
2.  **Iterative Counting:** The algorithm scans the dataset to count frequencies of *consecutive pairs only*. Non-consecutive pairings are ignored because text can only be compressed by collapsing tokens that are physically touching.
3.  **Greedy Merging:** The single most frequent adjacent pair is declared the "winner," merged into a new subword token, and assigned the next available unique integer ID.
4.  **Termination:** This loop repeats iteratively until either a pre-defined target `vocab_size` is reached or no more valid pairs exist to merge.
5.  **Longest-Match Principle (Encoding):** During active text encoding, the trained vocabulary is sorted by string length in descending order. This forces the tokenizer to always prioritize matching the largest possible subword blocks first, ensuring maximal compression.
"""