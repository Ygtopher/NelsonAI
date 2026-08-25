import os
import random
from datetime import datetime, timedelta
import subprocess

NUM_COMMITS = 1000

print("==================================================")
print("  🚀 GITHUB GREEN GRAPH BOOSTER (REALISTIC MODE)")
print("==================================================")

dummy_file = "research_engine.py"

commit_messages = [
    "Refactor attention mechanism for faster CUDA execution",
    "Optimize memory allocation in forward pass",
    "Update learning rate scheduler to CosineAnnealing",
    "Fix tensor shape mismatch in cross-entropy loss",
    "Implement Grouped Query Attention (GQA)",
    "Clean up tokenizer vocabulary mapping",
    "Tweak mixed precision (fp16) gradient scaler",
    "Add dropout layers to prevent overfitting",
    "Restructure dataset batching logic",
    "Fix NaN loss spike during warmup steps",
    "Enhance tool-calling routing logic",
    "Update model hyperparameters for 100M scale",
    "Implement KV caching for faster inference",
    "Fix self-evolution web scraping timeouts",
    "Add logging for perplexity metrics",
]

ALL_LINES = [
    "import torch",
    "import torch.nn as nn",
    "import torch.nn.functional as F",
    "import math",
    "class Attention(nn.Module):",
    "    def __init__(self, d_model):",
    "        super().__init__()",
    "        self.d_model = d_model",
    "        self.q_proj = nn.Linear(d_model, d_model)",
    "        self.k_proj = nn.Linear(d_model, d_model)",
    "        self.v_proj = nn.Linear(d_model, d_model)",
    "    def forward(self, x):",
    "        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)",
    "        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)",
    "        return F.softmax(scores, dim=-1)",
    "def calculate_loss(logits, targets):",
    "    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))",
    "optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)",
    "scaler = torch.cuda.amp.GradScaler()",
    "for epoch in range(epochs):",
    "    optimizer.zero_grad()",
    "    with torch.cuda.amp.autocast():",
    "        outputs = model(inputs)",
    "    loss = calculate_loss(outputs, labels)",
    "    scaler.scale(loss).backward()",
    "    scaler.step(optimizer)",
    "    scaler.update()",
    "def apply_rotary_embeddings(q, k):",
    "    return q, k  # Placeholder for RoPE",
    "# TODO: Implement FlashAttention for context > 2048",
    "vocab_size = 32000",
    "hidden_dim = 768",
    "num_layers = 12",
    "print(f'Training step {step} - Loss: {loss.item():.4f}')",
    "model = Nelson(config).to(device)",
    "model.eval()",
    "with torch.no_grad():",
    "    logits, _ = model(ctx)",
    "    logits = logits[:, -1, :] / temperature",
]

# Create/Clear the file
with open(dummy_file, "w") as f:
    f.write("# Nelson AI Core Research Engine\n\n")

for i in range(NUM_COMMITS):
    # Random date in the past 12 months
    days_ago = random.randint(0, 360)
    random_date = datetime.now() - timedelta(days=days_ago)
    git_date = random_date.strftime("%a, %d %b %Y %H:%M:%S %z")
    
    # Select 4 random lines of code to simulate a robust commit
    selected_lines = random.choices(ALL_LINES, k=4)
    snippet = "".join([f"{line}\n" for line in selected_lines])
    
    msg = random.choice(commit_messages)
    
    # Write to file
    with open(dummy_file, "a") as f:
        f.write(snippet)
        
    # Add and Commit
    subprocess.run(["git", "add", "-f", dummy_file], stdout=subprocess.DEVNULL)
    
    env = os.environ.copy()
    env["GIT_AUTHOR_DATE"] = git_date
    env["GIT_COMMITTER_DATE"] = git_date
    
    subprocess.run(
        ["git", "commit", "-m", f"{msg} (v{i})"],
        env=env,
        stdout=subprocess.DEVNULL
    )
    
    if i % 100 == 0:
        print(f"✓ Generated {i} realistic commits...")

print(f"\n✅ Successfully generated {NUM_COMMITS} realistic commits!")
print("You can now run: git push -u origin main --force")
