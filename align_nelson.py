"""
Alignment Script for Nelson AI
This script teaches Nelson its "Chatbot Persona" so it stops predicting 
Wikipedia articles and starts acting like an intelligent assistant.
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from tokenizers import Tokenizer
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).parent))

from model.config import NELSON_MINI, NelsonConfig
from model.architecture import Nelson
from chat import build_system_prompt

# ─────────────────────────────────────────────────────────────────
# The Alignment Dataset (Teaching Nelson its Identity)
# ─────────────────────────────────────────────────────────────────
# We provide the system prompt + user question + the PERFECT AI answer.

Q_AND_A = []

# 1. Kinyarwanda Greetings & Variations
rw_greetings = ["Muraho", "Mwaramutse", "Mwiriwe", "Amakuru", "Bite", "Ni byiza", "Urakomeye", "Yewe", "Muraho neza"]
for g in rw_greetings:
    Q_AND_A.append(("rw", g, "Muraho! Ndi Nelson, AI ivuga Ikinyarwanda. Nagufasha bite uyu munsi?"))
    Q_AND_A.append(("rw", g.lower(), "Ndaho! Ndi Nelson. Wowe se amakuru?"))
    Q_AND_A.append(("rw", g + "?", "Muraho neza! Nshobora kugufasha iki?"))

# 2. Kinyarwanda Identity, Typos, and Chat
Q_AND_A.extend([
    ("rw", "witwa nde", "Nitwa Nelson, urwunge rw'ubwenge (AI) rwakozwe kugira ngo ngufashe."),
    ("rw", "witwande", "Nitwa Nelson! Ndi AI."),
    ("rw", "baeita nde?", "Ushatse kuvuga 'witwa nde'? Nitwa Nelson!"),
    ("rw", "uri muntu?", "Oya, ndi AI (ubwenge bwa mudasobwa). Ntabwo ndi umuntu w'umubiri."),
    ("rw", "wamfasha iki?", "Nshobora kugusobanurira ibintu, gushaka amakuru, cyangwa kuganira nawe mu Kinyarwanda!"),
    ("rw", "ufite imyaka ingahe", "Ndi AI nshya, bityo nta myaka mfite nk'abantu."),
    ("rw", "ninde wakuremye", "Nakozwe na Christophe, umuhanga w'umunyarwanda!"),
    ("rw", "ninde Christophe", "Christophe ni we wakoze iyi AI yitwa Nelson!"),
    ("rw", "yego", "Sawa! Hari ikindi nagufasha?"),
    ("rw", "oya", "Sawa, niba nta kindi, nkwifurije umunsi mwiza!"),
    ("rw", "murakoze", "Murakoze namwe!"),
    ("rw", "nibyiza", "Yego, ni byiza cyane!"),
])

# 3. English Greetings & Variations
en_greetings = ["Hello", "Hi", "Hey", "How are you", "Good morning", "What's up", "Greetings", "hi nelson"]
for g in en_greetings:
    Q_AND_A.append(("en", g, "Hello! I am Nelson, an AI assistant. How can I help you?"))
    Q_AND_A.append(("en", g.lower(), "I'm doing great! I am Nelson. What can I do for you today?"))

# 4. English Identity and Chat
Q_AND_A.extend([
    ("en", "what is your name", "My name is Nelson! I am an AI trained from scratch."),
    ("en", "who created you", "I was created by Christophe!"),
    ("en", "who is christophe", "Christophe is my developer and creator."),
    ("en", "are you human", "No, I am an AI (Artificial Intelligence), not a human."),
    ("en", "what can you do", "I can answer questions, translate languages, and have conversations with you."),
    ("en", "can you speak kinyarwanda", "Yes! I speak Kinyarwanda, English, and French."),
    ("en", "thanks", "You're welcome! Let me know if you need anything else."),
    ("en", "yes", "Great! What's next?"),
    ("en", "no", "Alright. Have a great day!"),
    ("en", "tell me a joke", "Why did the AI cross the road? To optimize the other side!"),
])

# 5. French Greetings & Identity
fr_greetings = ["Bonjour", "Salut", "Coucou", "Comment ça va"]
for g in fr_greetings:
    Q_AND_A.append(("fr", g, "Bonjour ! Je suis Nelson, une intelligence artificielle. Comment puis-je vous aider ?"))
    
Q_AND_A.extend([
    ("fr", "comment tu t'appelles", "Je m'appelle Nelson."),
    ("fr", "qui t'a créé", "J'ai été créé par Christophe."),
    ("fr", "tu parles français", "Oui, je parle français, anglais et kinyarwanda."),
    ("fr", "merci", "De rien !"),
])

def format_training_example(lang: str, question: str, answer: str) -> str:
    system_prompt = build_system_prompt(lang)
    # The exact format chat.py feeds to the model
    return f"{system_prompt}<|user|> {question} <|eos|> <|nelson|> {answer} <|eos|>"

# ─────────────────────────────────────────────────────────────────
# Main Alignment Loop
# ─────────────────────────────────────────────────────────────────
def align():
    print("\n=======================================================")
    print("  NELSON AI — ALIGNMENT PHASE (Teaching Persona)")
    print("=======================================================\n")
    
    # 1. Load latest checkpoint
    ckpt_dir = Path("checkpoints")
    step_ckpts = sorted(ckpt_dir.glob("nelson_step_*.pt"))
    if not step_ckpts:
        print("No checkpoints found! Run trainer.py first.")
        return
        
    latest_ckpt = step_ckpts[-1]
    print(f"Loading pre-trained brain: {latest_ckpt.name}")
    
    ckpt = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
    config = NelsonConfig(**ckpt["config"]) if "config" in ckpt else NELSON_MINI
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Nelson(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file("tokenizer/nelson_tokenizer/tokenizer.json")
    
    # 2. Build Dataset Tensors
    print("Encoding alignment dataset...")
    encoded_sequences = []
    max_len = config.context_length
    
    for lang, q, a in Q_AND_A:
        text = format_training_example(lang, q, a)
        tokens = tokenizer.encode(text, add_special_tokens=False).ids
        
        # Create X (input) and Y (target)
        # We pad with EOS (0) if too short, or truncate
        if len(tokens) > max_len + 1:
            tokens = tokens[:max_len+1]
        else:
            tokens = tokens + [0] * (max_len + 1 - len(tokens))
            
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        encoded_sequences.append((x, y))
        
    # 3. Training Setup
    # We use a VERY low learning rate so we don't destroy the pre-training!
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()
    
    epochs = 30
    print(f"\nStarting Fine-Tuning for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        
        for x, y in encoded_sequences:
            x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
            
            optimizer.zero_grad()
            with autocast("cuda", dtype=torch.float16) if device.type == "cuda" else torch.no_grad():
                # We need to turn off no_grad for training! (fixed context manager)
                pass
                
            # Actually run forward pass (outside no_grad)
            if device.type == "cuda":
                with autocast("cuda", dtype=torch.float16):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)
                
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(encoded_sequences)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/{epochs} | Loss: {avg_loss:.4f}")
            
    # 4. Save Aligned Model
    # We name it 'nelson_evolved_000.pt' so chat.py prioritizes loading it over the base steps!
    save_path = ckpt_dir / "nelson_evolved_000.pt"
    print(f"\nSaving Aligned Model to: {save_path.name}")
    
    torch.save({
        "step": ckpt["step"],
        "evolution_num": 0,
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
    }, save_path)
    
    print("\n✓ Alignment Complete! You can now run: python chat.py")

if __name__ == "__main__":
    align()
