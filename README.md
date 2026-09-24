# Mini

A small English chatbot trained from scratch. It is a decoder-only transformer: 6 layers, 4 attention heads, embedding size 192, context of 512 tokens, and a vocabulary of 507 words. It answers short questions from its training dialogues and can tell two short stories. It is not a general-purpose assistant.

The chat window is at http://127.0.0.1:8765. The top shows a model graph and, for each reply, a star of the words the model looked at. Click **vocabulary** to open the word list. The chat stays at the bottom.

## Run

```bash
pip install -r requirements.txt
python serve.py
```

Then open http://127.0.0.1:8765. The server uses the NVIDIA GPU when CUDA is available, and the CPU otherwise. The trained weights are in `checkpoints/model.pt`.

`python chat.py` is the same model in the terminal.

## Train

`python train.py` reads `data/dialogues.txt` and `data/stories.txt`, trains for 2000 steps, and overwrites `checkpoints/model.pt`.

## Hugging Face

The Hub copy is in `hub/`. Rebuild it after a new training run with `python export_hub.py`.

The published model is [StrongDev2024/mini](https://huggingface.co/StrongDev2024/mini).

```python
from model import TinyGPT
from tokenizer import Tokenizer

model = TinyGPT.from_pretrained("StrongDev2024/mini")
tokenizer = Tokenizer.from_pretrained("StrongDev2024/mini")
```

Prompts look like `<user> what is 2 + 2 <bot>`. Generation stops at `<end>`.

## License

MIT
