# Nelson AI Core Research Engine

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
import torch.nn as nn
class Attention(nn.Module):
scaler = torch.cuda.amp.GradScaler()
        outputs = model(inputs)
import torch.nn.functional as F
    logits = logits[:, -1, :] / temperature
    scaler.step(optimizer)
print(f'Training step {step} - Loss: {loss.item():.4f}')
        super().__init__()
        self.d_model = d_model
        return F.softmax(scores, dim=-1)
