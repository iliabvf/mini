"""Write the Hugging Face folder from checkpoints/model.pt.

The folder is hub/. It holds the weights, the vocabulary, and the code
needed to load Mini with from_pretrained.
"""

import shutil
from pathlib import Path

from chat import load_checkpoint

ROOT = Path(__file__).resolve().parent
HUB = ROOT / "hub"
CODE_FILES = ("model.py", "tokenizer.py", "pretrained.py")


def main():
    model, tokenizer = load_checkpoint("cpu")
    model.save_pretrained(HUB)
    tokenizer.save_pretrained(HUB)
    for name in CODE_FILES:
        shutil.copy(ROOT / name, HUB / name)
    print(f"wrote {HUB}")


if __name__ == "__main__":
    main()
