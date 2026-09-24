"""Turn text into token ids and token ids back into text.

The model never sees letters. It sees integers. This file builds that
mapping from the training dialogues.
"""

import json
import re
from pathlib import Path

from pretrained import resolve_pretrained_folder

# Reserved tokens. They are written into the dialogue file and into chat prompts.
# <pad> evens out lengths if a batch needs filler.
# <unk> stands for a word that was not in the training file.
SPECIAL_TOKENS = ["<pad>", "<unk>", "<user>", "<bot>", "<end>"]

# Special tokens are kept whole. Other text splits into words and punctuation.
TOKEN_RE = re.compile(r"<pad>|<unk>|<user>|<bot>|<end>|\w+|[^\w\s]", re.UNICODE)


def tokenize(text):
    """Split lowercased text into a list of token strings."""
    return TOKEN_RE.findall(text.lower())


class Tokenizer:
    def __init__(self, token_to_id):
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}
        self.unk_id = self.token_to_id["<unk>"]

    @classmethod
    def build(cls, text):
        """Assign an id to every special token, then to every token in the text."""
        token_to_id = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        for token in tokenize(text):
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)
        return cls(token_to_id)

    def encode(self, text):
        """Text to a list of ids. Unknown words become <unk>."""
        return [self.token_to_id.get(token, self.unk_id) for token in tokenize(text)]

    def decode(self, ids):
        """Ids to a readable string. Special tokens are left out of the reply."""
        words = []
        for token_id in ids:
            token = self.id_to_token.get(int(token_id), "<unk>")
            if token in SPECIAL_TOKENS:
                continue
            words.append(token)
        text = " ".join(words)
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        return text.strip()

    def save_pretrained(self, folder):
        """Write vocab.json and tokenizer_config.json for the Hub."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "vocab.json").write_text(
            json.dumps(self.token_to_id, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        meta = {
            "tokenizer_class": "Tokenizer",
            "lowercase": True,
            "special_tokens": SPECIAL_TOKENS,
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        }
        (folder / "tokenizer_config.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_pretrained(cls, path_or_repo, **_ignored):
        """Load the word vocabulary from a local hub folder or a Hub repo id."""
        folder = resolve_pretrained_folder(path_or_repo)
        raw = json.loads((folder / "vocab.json").read_text(encoding="utf-8"))
        token_to_id = {token: int(index) for token, index in raw.items()}
        return cls(token_to_id)
