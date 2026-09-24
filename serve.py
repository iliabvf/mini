"""Open the Mini window: thinking picture on top, chat underneath.

Each generated word is one event. The event carries the word the model
chose, the other words it almost chose, and how much each head looked
at the tokens already on the page.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import torch.nn.functional as F

from chat import FALLBACK, MAX_NEW_TOKENS, first_sentence, load_checkpoint, prompt_ids

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
HOST = "127.0.0.1"
PORT = 8765
TOP_K = 8
STORY_SENTENCES = 4
STORY_TOKENS = 90
SENTENCE_ENDINGS = {".", "?", "!"}

device = "cuda" if torch.cuda.is_available() else "cpu"
model, tokenizer = load_checkpoint(device)
think_lock = threading.Lock()
PARAM_COUNT = sum(parameter.numel() for parameter in model.parameters())


def token_label(token_id):
    return tokenizer.id_to_token.get(int(token_id), "<unk>")


def clean_history(raw):
    """Keep the last four finished turns sent by the page."""
    turns = []
    if isinstance(raw, list):
        for item in raw[-4:]:
            if not isinstance(item, dict):
                continue
            user = str(item.get("user", "")).strip()
            bot = str(item.get("bot", "")).strip()
            if user and bot:
                turns.append((user, bot))
    return turns


def model_profile():
    """Counts the model can draw: size, shape, and where the parameters sit."""
    layers = model.config["n_layer"]
    buckets = {"embedding": 0, "attention": 0, "mlp": 0, "norm": 0, "output": 0}
    per_layer = [0] * layers
    for name, param in model.named_parameters():
        count = int(param.numel())
        if name.startswith("blocks."):
            per_layer[int(name.split(".")[1])] += count
            if ".attn." in name:
                buckets["attention"] += count
            elif ".mlp." in name:
                buckets["mlp"] += count
            else:
                buckets["norm"] += count
        elif name.startswith("tok_emb") or name.startswith("pos_emb"):
            buckets["embedding"] += count
        elif name.startswith("head"):
            buckets["output"] += count
        else:
            buckets["norm"] += count
    return {
        "model": "Mini",
        "parameters": PARAM_COUNT,
        "layers": layers,
        "heads": model.config["n_head"],
        "embedding": model.config["n_embd"],
        "vocabulary": len(tokenizer.token_to_id),
        "words": [tokenizer.id_to_token[index] for index in range(len(tokenizer.token_to_id))],
        "parts": [
            {"name": name, "parameters": buckets[name]}
            for name in ("embedding", "attention", "mlp", "norm", "output")
        ],
        "perLayer": per_layer,
    }


def attach_stats(event, prompt_len, generated, window_len):
    """Parameter totals stay fixed. Token counts change with each chosen word."""
    event["stats"] = {
        "parameters": PARAM_COUNT,
        "layers": model.config["n_layer"],
        "heads": model.config["n_head"],
        "embedding": model.config["n_embd"],
        "context": model.block_size,
        "vocab": len(tokenizer.token_to_id),
        "prompt": prompt_len,
        "generated": generated,
        "window": window_len,
    }


def is_story(user_text):
    """A story request may run for several sentences. Other questions stop at <end>."""
    lowered = user_text.lower()
    return any(phrase in lowered for phrase in ("story", "what happened", "tell me more", "go on"))


def think(user_text, history):
    """Yield one JSON-ready step per chosen token, then a finished reply."""
    story = is_story(user_text)
    ids = prompt_ids(tokenizer, history, user_text, model.block_size)
    end_id = tokenizer.token_to_id["<end>"]
    user_id = tokenizer.token_to_id["<user>"]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    reply_ids = []
    sentence_ends = 0
    token_limit = STORY_TOKENS if story else MAX_NEW_TOKENS

    for _ in range(token_limit):
        window = idx[:, -model.block_size :]
        logits, _loss, attentions = model(window, return_attn=True)
        probs = F.softmax(logits[0, -1], dim=-1)
        next_id = int(torch.argmax(probs).item())
        top_values, top_indices = torch.topk(probs, k=min(TOP_K, probs.numel()))
        context_ids = window[0].tolist()
        layers = []
        for weights in attentions:
            heads = weights[0, :, -1, :].tolist()
            layers.append([[round(float(value), 4) for value in head] for head in heads])
        label = token_label(next_id)
        step = {
            "type": "step",
            "chosen": label,
            "context": [token_label(token_id) for token_id in context_ids],
            "probs": [
                {"token": token_label(int(index)), "p": round(float(value), 4)}
                for value, index in zip(top_values, top_indices)
            ],
            "layers": layers,
            "reply": tokenizer.decode(reply_ids),
        }
        if next_id == user_id or (next_id == end_id and not story):
            step["stop"] = True
            attach_stats(step, len(ids), len(reply_ids), len(context_ids))
            yield step
            break
        if next_id == end_id and sentence_ends >= STORY_SENTENCES:
            step["stop"] = True
            attach_stats(step, len(ids), len(reply_ids), len(context_ids))
            yield step
            break
        if next_id == end_id:
            continue
        reply_ids.append(next_id)
        step["reply"] = tokenizer.decode(reply_ids)
        attach_stats(step, len(ids), len(reply_ids), len(context_ids))
        yield step
        idx = torch.cat([idx, torch.tensor([[next_id]], device=device)], dim=1)
        if story and label in SENTENCE_ENDINGS:
            sentence_ends += 1
            if sentence_ends >= STORY_SENTENCES:
                break
    else:
        text = tokenizer.decode(reply_ids)
        if not story:
            text = first_sentence(text)
        done = {"type": "done", "reply": text or FALLBACK}
        attach_stats(done, len(ids), len(reply_ids), min(len(ids) + len(reply_ids), model.block_size))
        yield done
        return
    done = {"type": "done", "reply": tokenizer.decode(reply_ids) or FALLBACK}
    attach_stats(done, len(ids), len(reply_ids), min(len(ids) + len(reply_ids), model.block_size))
    yield done


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"{self.address_string()} {fmt % args}")

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/model":
            body = json.dumps(model_profile()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path != "/":
            self.send_error(404)
            return
        page = (WEB / "index.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/chat":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            payload = {}
        message = str(payload.get("message", "")).strip()
        history = clean_history(payload.get("history"))
        if not message:
            body = json.dumps({"error": "empty message"}).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if not think_lock.acquire(blocking=False):
            body = json.dumps({"error": "busy"}).encode("utf-8")
            self.send_response(409)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            for event in think(message, history):
                self.wfile.write((json.dumps(event) + "\n").encode("utf-8"))
                self.wfile.flush()
        finally:
            think_lock.release()


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Mini is at http://{HOST}:{PORT}")
    print(f"device: {device}  layers: {model.config['n_layer']}  heads: {model.config['n_head']}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
