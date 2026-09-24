"""A small decoder-only transformer.

It reads tokens from left to right and scores the next token.
Causal attention means a position may look at earlier tokens only.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from pretrained import resolve_pretrained_folder

# Keys TinyGPT.__init__ accepts. config.json may also carry Hub metadata.
CONFIG_KEYS = ("vocab_size", "block_size", "n_layer", "n_head", "n_embd", "dropout")


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        # Lower triangle is 1: each token may attend to itself and the past.
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x, return_attn=False):
        batch, time, channels = x.shape
        qkv = self.qkv(x)
        query, key, value = qkv.split(channels, dim=2)
        query = query.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        key = key.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        value = value.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        scores = (query @ key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(self.mask[:, :, :time, :time] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        mixed = (self.dropout(weights) @ value).transpose(1, 2).contiguous().view(batch, time, channels)
        out = self.dropout(self.proj(mixed))
        if return_attn:
            return out, weights
        return out


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_attn=False):
        attended = self.attn(self.ln1(x), return_attn=return_attn)
        if return_attn:
            attended, weights = attended
        x = x + attended
        x = x + self.mlp(self.ln2(x))
        if return_attn:
            return x, weights
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size=128, n_layer=2, n_head=4, n_embd=128, dropout=0.1):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.config = {
            "vocab_size": vocab_size,
            "block_size": block_size,
            "n_layer": n_layer,
            "n_head": n_head,
            "n_embd": n_embd,
            "dropout": dropout,
        }

    def forward(self, idx, targets=None, return_attn=False):
        _batch, time = idx.shape
        positions = torch.arange(time, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(positions))
        attentions = []
        for block in self.blocks:
            if return_attn:
                x, weights = block(x, return_attn=True)
                attentions.append(weights)
            else:
                x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if return_attn:
            return logits, loss, attentions
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.0, top_k=None, stop_ids=None):
        """Append tokens until a stop token or the length limit.

        temperature 0 always picks the most likely next token.
        """
        stop_ids = set(stop_ids or [])
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-6)
                if top_k is not None:
                    top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = logits.masked_fill(logits < top_values[:, [-1]], float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if int(next_id.item()) in stop_ids:
                break
        return idx

    def save_pretrained(self, folder):
        """Write config.json and model.safetensors for the Hugging Face Hub."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        payload = {"model_type": "tiny-gpt", **self.config}
        (folder / "config.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        from safetensors.torch import save_file

        state = {name: value.detach().cpu().contiguous() for name, value in self.state_dict().items()}
        save_file(state, str(folder / "model.safetensors"))

    @classmethod
    def from_pretrained(cls, path_or_repo, **_ignored):
        """Load Mini from a local hub folder or a Hugging Face repo id.

        trust_remote_code is accepted and ignored. Import this class from
        model.py, then call from_pretrained with the repo id.
        """
        folder = resolve_pretrained_folder(path_or_repo)
        raw = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        missing = [key for key in CONFIG_KEYS if key not in raw]
        if missing:
            raise FileNotFoundError(f"{folder / 'config.json'} is missing {', '.join(missing)}")
        model = cls(**{key: raw[key] for key in CONFIG_KEYS})
        from safetensors.torch import load_file

        model.load_state_dict(load_file(str(folder / "model.safetensors")))
        model.eval()
        return model
