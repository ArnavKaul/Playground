# Parameter-Efficient Fine-Tuning (PEFT) Sandbox: NLLB-200 for Catalan Translation

## Project Overview
This repository serves as a practical, lightweight sandbox demonstrating **Parameter-Efficient Fine-Tuning (PEFT)** using **Low-Rank Adaptation (LoRA)**. The pipeline takes Meta's `nllb-200-distilled-600M` model and parameter-efficiently customizes it to translate English text into Catalan using the `Helsinki-NLP/opus_books` dataset. It culminates in generating an explicit computation graph visualization of the model's forward pass logits using `torchviz`.

---

## Core Concept: What is PEFT & LoRA?

### Why PEFT?
[cite_start]Full-parameter fine-tuning of Large Language Models (LLMs) or large conditional sequence-to-sequence models (like NLLB) requires updating billions of weights[cite: 15]. This introduces severe engineering and resource constraints:
* [cite_start]**High Compute & Memory Costs:** Storing gradients, optimizer states (like AdamW's first and second moments), and weights for billions of parameters requires substantial GPU VRAM[cite: 15].
* **Catastrophic Forgetting:** Over-tuning a model on a narrow downstream task can wipe out its baseline multi-lingual text representation capabilities.
* **Storage Inefficiency:** Saving an entirely new copy of a multiple-gigabyte model checkpoint for every individual downstream task is unsustainable in production.

**PEFT** solves this by keeping the entire pre-trained backbone network completely frozen. It only injects or exposes a tiny fraction (< 1%) of specialized, trainable parameter modules.

### Low-Rank Adaptation (LoRA) Deep Dive
LoRA works on the foundational mathematical hypothesis that weight updates during adaptation have a low "intrinsic rank." 

Instead of modifying an original dense layer weight matrix $W_0 \in \mathbb{R}^{d \times k}$ directly, LoRA factors the required weight update $\Delta W$ into two low-rank matrices, $A$ and $B$:
$$\Delta W = B \times A$$
Where:
* $B \in \mathbb{R}^{d \times r}$
* $A \in \mathbb{R}^{r \times k}$
* The rank $r \ll \min(d, k)$

During the forward pass, the calculation splits symmetrically: the frozen baseline weights and the active low-rank adapters compute their states simultaneously and sum up:
$$h = W_0x + \Delta Wx = W_0x + \frac{\alpha}{r}(BA)x$$

---

## Implementation Details

Here is how the concepts above map directly into the provided Python pipeline:

### 1. Data Structuring & Domain Alignment
```python
def preprocess(translation):
    inputs = ["Translate to Catalan " + en for en in translation["translation"]]
    targets = [ row["ca"]   for row in translation["translation"] ]
    return tokenizer(inputs, text_target=targets, truncation=True, padding="max_length")
