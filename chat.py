"""Chat with the trained model in the terminal.

Your message is wrapped as <user> ... <bot>. The model then writes the
reply one token at a time. It always picks the most likely next word and
stops at <end>. If it misses that marker, the reply is cut at the first
finished sentence or at the start of the next user turn.
"""

from pathlib import Path

import torch

from model import TinyGPT
from tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent
CKPT_PATH = ROOT / "checkpoints" / "model.pt"
MAX_NEW_TOKENS = 40
FALLBACK = "i do not know that yet, ask me something simple."


def load_checkpoint(device):
    if not CKPT_PATH.is_file():
        raise SystemExit(f"missing {CKPT_PATH}. Train first with: python train.py")
    try:
        checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(CKPT_PATH, map_location=device)
    tokenizer = Tokenizer(checkpoint["vocab"])
    model = TinyGPT(**checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, tokenizer


def prompt_ids(tokenizer, history, user_text, block_size):
    """Build the token ids for past turns plus the new user line.

    Drop the oldest turns until the prompt fits in the context window.
    """
    bot_id = tokenizer.token_to_id["<bot>"]
    # Four earlier turns keep a short office chat on the same topic
    # and still leave room for the new question.
    turns = list(history)[-4:]
    while True:
        parts = [f"<user> {past_user} <bot> {past_bot} <end>" for past_user, past_bot in turns]
        parts.append(f"<user> {user_text} <bot>")
        ids = tokenizer.encode(" ".join(parts))
        if len(ids) <= block_size or not turns:
            break
        turns = turns[1:]
    if len(ids) > block_size:
        ids = ids[-(block_size - 1) :] + [bot_id]
    return ids


def without_repeats(ids):
    """Drop a phrase the model just said again.

    A short chunk copied immediately is a loop, not a new fact.
    Three identical tokens in a row are a loop too.
    """
    ids = list(ids)
    while ids:
        count = len(ids)
        cut = 0
        if count >= 3 and ids[-1] == ids[-2] == ids[-3]:
            cut = 1
        else:
            for size in range(2, 9):
                if count < size * 2:
                    break
                if ids[-size:] == ids[-2 * size : -size]:
                    cut = size
                    break
        if not cut:
            return ids
        ids = ids[:-cut]
    return ids


def first_sentence(text):
    """Keep text through the first period, question mark, or exclamation mark."""
    for index, char in enumerate(text):
        if char in ".!?":
            return text[: index + 1].strip()
    return text.strip()


def answer(model, tokenizer, history, user_text, device):
    ids = prompt_ids(tokenizer, history, user_text, model.block_size)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    end_id = tokenizer.token_to_id["<end>"]
    user_id = tokenizer.token_to_id["<user>"]
    generated = model.generate(
        idx,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
        stop_ids={end_id, user_id},
    )
    new_ids = generated[0, len(ids) :].tolist()
    if end_id in new_ids:
        new_ids = new_ids[: new_ids.index(end_id)]
    elif user_id in new_ids:
        new_ids = new_ids[: new_ids.index(user_id)]
    else:
        text = first_sentence(tokenizer.decode(without_repeats(new_ids)))
        return text or FALLBACK
    text = tokenizer.decode(without_repeats(new_ids))
    return text or FALLBACK


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_checkpoint(device)
    history = []
    print("Mini is ready. Type a message, or quit to stop.")
    while True:
        try:
            user_text = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_text.lower() in {"quit", "exit"}:
            break
        if not user_text:
            continue
        bot_text = answer(model, tokenizer, history, user_text, device)
        print(f"mini: {bot_text}")
        history.append((user_text, bot_text))
        history = history[-4:]


if __name__ == "__main__":
    main()
