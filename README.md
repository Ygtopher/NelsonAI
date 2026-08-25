# Nelson AI 🇷🇼

> ⚠️ **WORK IN PROGRESS:** This artificial intelligence model is currently in active development. It is still learning and undergoing alignment fine-tuning. Responses may be inaccurate, experimental, or incomplete.

**A Self-Evolving, Internet-Aware Trilingual AI — Trained From Scratch**

Nelson is a custom-built GPT-style transformer trained natively on **Kinyarwanda, English, and French**.  
It features live internet access via tool-calling and a self-evolution engine that continuously improves from new conversations and web data.

> **Language behavior:** Nelson detects which language you're using and responds in the same language.  
> It defaults to Kinyarwanda and has Kinyarwanda as its primary identity.

---

## 🚀 Full Training Pipeline

### Step 1 — Environment Setup (Python 3.11)
*Note: PyTorch CUDA requires Python 3.9 - 3.12. Ensure you are using Python 3.11!*
```powershell
# Create and activate a Python 3.11 virtual environment
py -3.11 -m venv venv
venv\Scripts\activate

# Install PyTorch with CUDA 12.1 (for Quadro T2000)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

### Step 2 — Download Multilingual Data
Uses direct Parquet downloads and streaming to pull high-quality data from Wikipedia, OPUS-100, Wikitext, and TinyStories.
```powershell
python data_collection/download_data.py
```

### Step 3 — Clean and Balance Corpus
Cleans text and strictly balances the final corpus to **60% Kinyarwanda, 25% English, 15% French**.
```powershell
python data_collection/clean_text.py
```

### Step 4 — Train the Tokenizer
Learns a custom 32,000-token BPE vocabulary covering all three languages.
```powershell
python tokenizer/train_tokenizer.py
```

### Step 5 — Tokenize the Dataset
Converts text into fast, binary `.npy` format for rapid training.
```powershell
python training/tokenize_dataset.py
```

### Step 6 — Train Nelson 🧠
Fully custom training loop with Mixed Precision (fp16), Gradient Accumulation, and checkpointing.
```powershell
# Start training (adjust batch size based on VRAM)
python training/trainer.py --batch-size 4
```

### Step 7 — Chat with Nelson 🗣️
Launch the trilingual terminal interface with internet capabilities.
```powershell
python chat.py
```

---

## 🧠 Architecture Profiles

The model size can be scaled in `model/config.py`. By default, it is training on the **NELSON_MINI** configuration.

| Component | `NELSON_NANO` (Testing) | `NELSON_MINI` (Production) |
|-----------|------------------------|---------------------------|
| **Parameters** | ~34.1 Million | **~100 Million (Tied)** |
| **Layers** | 6 | **12** |
| **d_model** | 512 | **768** |
| **Heads** | 8 (KV=4, GQA) | **12 (KV=4, GQA)** |
| **FFN dim** | 1408 (SwiGLU) | **2048 (SwiGLU)** |
| **Context** | 512 tokens | **512 tokens** |
| **Vocab** | 32,000 (BPE) | **32,000 (BPE)** |
| **VRAM Cost** | ~68 MB | **~200 MB** |

*Both models use RoPE positional encoding, RMSNorm, SwiGLU activations, and Tied Embeddings for maximum efficiency on the Quadro T2000.*

---

## 🌐 Internet Access

Nelson can search the internet mid-conversation using tool tokens it learns during fine-tuning. The chat router intercepts these tokens automatically:

```text
<|search|>ibibazo byawe<|/search|>    → Web search (DuckDuckGo)
<|wiki|>ikibazo<|/wiki|>              → Wikipedia (Kinyarwanda/English/French)
<|fetch|>https://...<|/fetch|>        → Read any specific webpage
```

---

## ⚡ Self-Evolution Engine

Nelson improves itself automatically over time:

1. **Conversation Logging:** Every chat is saved to memory.
2. **Web Scraping:** It periodically collects fresh Kinyarwanda content from the web to fill gaps in its knowledge.
3. **Continuous Fine-Tuning:** It runs a lightweight training loop in the background to permanently memorize new facts without forgetting old ones.
4. **Hot-Swapping:** Generates new model weights (e.g., `nelson_evolved_001.pt`).

*(Evolution triggers automatically in the background every 5 conversations).*

---

## 💻 Terminal Chat Commands

While running `python chat.py`, you can use the following commands:

| Command | Description |
|---------|-------------|
| `/search <q>` | Force a manual web search |
| `/wiki <topic>` | Force a Wikipedia search |
| `/fetch <url>` | Read the contents of a URL |
| `/evolve` | Trigger self-evolution right now |
| `/stats` | Show memory & evolution stats |
| `/temp 0.7` | Set creativity temperature (0.1=focused, 1.5=wild) |
| `/settings` | View all active generation settings |
| `/quit` | Save memory and exit |

---

## 🖥️ Hardware Profile
Built and optimized specifically for edge hardware constraints:
- **GPU:** NVIDIA Quadro T2000 (4GB VRAM)
- **RAM:** 16GB
- **Storage:** 90GB free space
