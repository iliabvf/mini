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
BUSINESS_PATH = DATA_DIR / "business.txt"
BUSINESS_REPEATS = 4
DRILL_FRACTION = 0.8
CKPT_PATH = ROOT / "checkpoints" / "model.pt"

BLOCK_SIZE = 512
N_LAYER = 6
N_HEAD = 4
N_EMBD = 192
DROPOUT = 0.0
BATCH_SIZE = 4
LEARNING_RATE = 3e-4
STEPS = 6000


def read_corpus():
    """Dialogues teach replies. Stories teach the model to continue a sentence.

    The business file is repeated so office words are learned along with
    the shorter chats already in the dialogue file.
    """
    parts = []
    for path in (DIALOGUES_PATH, STORIES_PATH, BUSINESS_PATH):
        if not path.is_file():
            raise SystemExit(f"missing training file: {path}")
        parts.append(path.read_text(encoding="utf-8"))
    business = parts.pop()
    parts.append("\n".join([business] * BUSINESS_REPEATS))
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


def parse_pairs(text):
    """Pull each user and bot line out of a dialogue file."""
    pairs = []
    for block in text.split("\n\n"):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) >= 3 and lines[0].startswith("<user> ") and lines[1].startswith("<bot> "):
            pairs.append((lines[0][len("<user> ") :], lines[1][len("<bot> ") :]))
    return pairs


# Four-turn office chats. Each step is trained with the turns before it,
# so a later question still knows the invoice, the price, or the meeting.
CONVERSATIONS = [
    [
        ("please email the client about the invoice", "i will email the client and attach the invoice."),
        ("what will you say", "i will say the invoice is attached and payment is due on friday."),
        ("the client did not pay", "i will send a short reminder."),
        ("what should the reminder say", "the reminder should say the payment is still due on friday."),
    ],
    [
        ("please send a quote for this order", "i will send a quote with the price and the terms today."),
        ("that price is high", "we can offer a small discount if you order today."),
        ("we can order more", "a larger order can take a larger discount."),
        ("let us agree", "we agree on the lower price, with payment on friday and delivery on monday."),
    ],
    [
        ("let us meet on monday", "yes. we can meet on monday at ten."),
        ("what should we discuss", "we should discuss the budget, the deadline, and the client proposal."),
        ("monday is not possible", "we can move the meeting to thursday at ten."),
        ("please confirm that", "confirmed. the meeting is on thursday at ten."),
    ],
    [
        ("a customer wrote about a broken item", "i will read the complaint and check the order."),
        ("what is wrong with the item", "the item arrived broken."),
        ("should the customer get a refund", "we should refund the customer for the broken item."),
        ("please send that refund", "i will send the refund to the customer today."),
    ],
]


def drill_texts(dialogues, business):
    """One exact reply per question, plus each step of a longer chat.

    Each window holds many different replies, so a salary line is not
    copied over and over beside the wage line.
    """
    seen = set()
    texts = []

    def add(text):
        if text not in seen:
            seen.add(text)
            texts.append(text)

    focus = {
        "what is a meeting",
        "what is a resignation",
        "let us agree",
        "what is an appointment",
        "what is an executive",
    }
    for user, bot in parse_pairs(dialogues) + parse_pairs(business):
        text = f"<user> {user} <bot> {bot} <end>"
        add(text)
        if user in focus:
            texts.extend([text] * 11)
    for convo in CONVERSATIONS:
        parts = []
        for user, bot in convo:
            parts.append(f"<user> {user} <bot> {bot} <end>")
            add(" ".join(parts))
    return texts


def build_drills(tokenizer, texts):
    """Pack different replies into each window.

    The same short answer is not copied across the window. That copy
    taught the model to repeat a phrase.
    """
    encoded = []
    for text in texts:
        ids = tokenizer.encode(text)
        if len(ids) >= 2:
            encoded.append(ids)
    if not encoded:
        return None
    rows = []
    width = BLOCK_SIZE + 1
    count = len(encoded)
    for start in range(count):
        seq = []
        step = 0
        while len(seq) < width and step < count:
            nxt = encoded[(start + step) % count]
            step += 1
            if seq and nxt == seq[-len(nxt) :]:
                continue
            seq.extend(nxt)
        if len(seq) < 2:
            continue
        if len(seq) < width:
            mixed = list(seq)
            while len(seq) < width:
                seq.extend(mixed)
        rows.append(seq[:width])
    return torch.tensor(rows, dtype=torch.long)


def get_batch(stream, drills):
    """Pick random windows. The target is the input shifted one token ahead.

    Most windows are a single repeated reply, so close definitions stay apart.
    The rest are ordinary text, which keeps the stories and the chat flow.
    """
    if drills is not None and torch.rand(1).item() < DRILL_FRACTION:
        choice = torch.randint(0, drills.size(0), (BATCH_SIZE,))
        batch = drills[choice]
        return batch[:, :-1], batch[:, 1:]
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
    drills = build_drills(
        tokenizer,
        drill_texts(
            DIALOGUES_PATH.read_text(encoding="utf-8"),
            BUSINESS_PATH.read_text(encoding="utf-8"),
        ),
    )

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
    print(f"drills: {0 if drills is None else drills.size(0)}")

    model.train()
    for step in range(1, STEPS + 1):
        inputs, targets = get_batch(stream, drills)
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
