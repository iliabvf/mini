---
license: mit
language:
- en
library_name: pytorch
tags:
- text-generation
- chatbot
---

# Mini

A small English chatbot trained from scratch. It is a decoder-only transformer with 6 layers, 4 attention heads, an embedding size of 192, and a context window of 512 tokens. The vocabulary is 980 words, including a business English word list. The chat keeps the last four turns. It answers short questions it has seen in its training dialogues. It is not a general-purpose assistant.

## Load

`model.py`, `tokenizer.py`, and `pretrained.py` from this repo need to be on the Python path.

```python
from model import TinyGPT
from tokenizer import Tokenizer

model = TinyGPT.from_pretrained("StrongDev2024/mini", trust_remote_code=True)
tokenizer = Tokenizer.from_pretrained("StrongDev2024/mini")

ids = tokenizer.encode("<user> what is 2 + 2 <bot>")
import torch
out = model.generate(
    torch.tensor([ids]),
    max_new_tokens=40,
    temperature=0.0,
    stop_ids={tokenizer.token_to_id["<end>"], tokenizer.token_to_id["<user>"]},
)
print(tokenizer.decode(out[0, len(ids) :].tolist()))
```

The prompt is `<user> your question <bot>`. Generation stops at `<end>`.

## License

MIT
