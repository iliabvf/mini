"""Train TinyGPT on dialogues and stories, then save checkpoints/model.pt.

Each step shows the model a chunk of that text and asks it to predict the
next token. The loss is how wrong those guesses are. Lower is better.
"""

from pathlib import Path

import torch

from model import TinyGPT
from tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DIALOGUES_PATH = DATA_DIR / "dialogues.txt"
STORIES_PATH = DATA_DIR / "stories.txt"
CKPT_PATH = ROOT / "checkpoints" / "model.pt"

BLOCK_SIZE = 512
N_LAYER = 6
N_HEAD = 4
N_EMBD = 192
DROPOUT = 0.0
BATCH_SIZE = 4
LEARNING_RATE = 3e-4
STEPS = 2000


def read_corpus():
    """Dialogues teach replies. Stories teach the model to continue a sentence."""
    parts = []
    for path in (DIALOGUES_PATH, STORIES_PATH):
        if not path.is_file():
            raise SystemExit(f"missing training file: {path}")
        parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def load_stream(tokenizer, text):
    """Encode the corpus and repeat it so random windows stay full."""
    ids = tokenizer.encode(text)
    if len(ids) < 2:
        raise SystemExit("the training files have no tokens")
    repeats = 8
    while len(ids) * repeats <= BLOCK_SIZE + 1:
        repeats *= 2
    return torch.tensor(ids * repeats, dtype=torch.long)


def get_batch(stream):
    """Pick random windows. The target is the input shifted one token ahead."""
    starts = torch.randint(0, len(stream) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    inputs = torch.stack([stream[start : start + BLOCK_SIZE] for start in starts])
    targets = torch.stack([stream[start + 1 : start + 1 + BLOCK_SIZE] for start in starts])
    return inputs, targets


def main():
    torch.manual_seed(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = read_corpus()
    tokenizer = Tokenizer.build(text)
    stream = load_stream(tokenizer, text)

    model = TinyGPT(
        vocab_size=len(tokenizer.token_to_id),
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=DROPOUT,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    print(f"device: {device}")
    print(f"model: {N_LAYER} layers, embedding {N_EMBD}, context {BLOCK_SIZE}")
    print(f"vocab: {len(tokenizer.token_to_id)} tokens")
    print(f"training tokens: {len(stream)}")

    model.train()
    for step in range(1, STEPS + 1):
        inputs, targets = get_batch(stream)
        inputs = inputs.to(device)
        targets = targets.to(device)
        _logits, loss = model(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == STEPS:
            print(f"step {step}/{STEPS}  loss {loss.item():.3f}")

    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": model.config,
            "vocab": tokenizer.token_to_id,
        },
        CKPT_PATH,
    )
    print(f"saved {CKPT_PATH}")
    print("next: python chat.py")


if __name__ == "__main__":
    main()
