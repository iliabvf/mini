"use strict";

const SPECIALS = ["<pad>", "<unk>", "<user>", "<bot>", "<end>"];
const TOKEN_RE = /<pad>|<unk>|<user>|<bot>|<end>|\w+|[^\w\s]/g;
const FALLBACK = "i do not know that yet, ask me something simple.";
const MAX_NEW_TOKENS = 40;
const STORY_TOKENS = 90;
const STORY_SENTENCES = 4;
const TOP_K = 8;

let words = [];
let tokenToId = new Map();
let tensors = new Map();
let nLayer = 6;
let nHead = 4;
let nEmbd = 192;
let blockSize = 512;
let vocab = 507;
let paramCount = 2962560;

function tokenize(text) {
  return (text.toLowerCase().match(TOKEN_RE) || []);
}

function encode(text) {
  const unk = tokenToId.get("<unk>");
  return tokenize(text).map((token) => (tokenToId.has(token) ? tokenToId.get(token) : unk));
}

function decode(ids) {
  const parts = [];
  ids.forEach((id) => {
    const token = words[id] || "<unk>";
    if (SPECIALS.includes(token)) return;
    parts.push(token);
  });
  return parts.join(" ").replace(/\s+([.,!?;:])/g, "$1").trim();
}

function promptIds(history, userText) {
  const botId = tokenToId.get("<bot>");
  let turns = (history || []).slice(-2);
  let ids = [];
  while (true) {
    const parts = turns.map((turn) => `<user> ${turn.user} <bot> ${turn.bot} <end>`);
    parts.push(`<user> ${userText} <bot>`);
    ids = encode(parts.join(" "));
    if (ids.length <= blockSize || !turns.length) break;
    turns = turns.slice(1);
  }
  if (ids.length > blockSize) ids = ids.slice(-(blockSize - 1)).concat([botId]);
  return ids;
}

function isStory(text) {
  const lowered = text.toLowerCase();
  return ["story", "what happened", "tell me more", "go on"].some((phrase) => lowered.includes(phrase));
}

function tensor(name) {
  const found = tensors.get(name);
  if (!found) throw new Error(`missing weight ${name}`);
  return found;
}

function layerNorm(x, rows, width, weight, bias) {
  const y = new Float32Array(x.length);
  const eps = 1e-5;
  for (let row = 0; row < rows; row++) {
    const offset = row * width;
    let mean = 0;
    for (let col = 0; col < width; col++) mean += x[offset + col];
    mean /= width;
    let variance = 0;
    for (let col = 0; col < width; col++) {
      const delta = x[offset + col] - mean;
      variance += delta * delta;
    }
    variance /= width;
    const scale = 1 / Math.sqrt(variance + eps);
    for (let col = 0; col < width; col++) {
      y[offset + col] = (x[offset + col] - mean) * scale * weight[col] + bias[col];
    }
  }
  return y;
}

function linear(x, rows, inDim, weight, bias, outDim) {
  const y = new Float32Array(rows * outDim);
  for (let row = 0; row < rows; row++) {
    const xOffset = row * inDim;
    const yOffset = row * outDim;
    for (let out = 0; out < outDim; out++) {
      const wOffset = out * inDim;
      let sum = bias ? bias[out] : 0;
      for (let col = 0; col < inDim; col++) sum += weight[wOffset + col] * x[xOffset + col];
      y[yOffset + out] = sum;
    }
  }
  return y;
}

function erf(x) {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const p = 0.3275911;
  const t = 1 / (1 + p * ax);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-ax * ax);
  return sign * y;
}

function gelu(value) {
  return 0.5 * value * (1 + erf(value / Math.SQRT2));
}

function attention(x, rows, layer) {
  const width = nEmbd;
  const heads = nHead;
  const headDim = width / heads;
  const scale = 1 / Math.sqrt(headDim);
  const qkv = linear(
    x,
    rows,
    width,
    tensor(`blocks.${layer}.attn.qkv.weight`),
    tensor(`blocks.${layer}.attn.qkv.bias`),
    width * 3,
  );
  const mixed = new Float32Array(rows * width);
  const last = Array.from({ length: heads }, () => new Array(rows).fill(0));
  for (let head = 0; head < heads; head++) {
    for (let row = 0; row < rows; row++) {
      const scores = new Float32Array(row + 1);
      let maxScore = -Infinity;
      for (let key = 0; key <= row; key++) {
        let dot = 0;
        for (let dim = 0; dim < headDim; dim++) {
          const query = qkv[row * width * 3 + head * headDim + dim];
          const keyValue = qkv[key * width * 3 + width + head * headDim + dim];
          dot += query * keyValue;
        }
        scores[key] = dot * scale;
        if (scores[key] > maxScore) maxScore = scores[key];
      }
      let total = 0;
      for (let key = 0; key <= row; key++) {
        scores[key] = Math.exp(scores[key] - maxScore);
        total += scores[key];
      }
      for (let key = 0; key <= row; key++) scores[key] /= total;
      if (row === rows - 1) {
        for (let key = 0; key < rows; key++) last[head][key] = Math.round(scores[key] * 10000) / 10000;
      }
      for (let dim = 0; dim < headDim; dim++) {
        let mixedValue = 0;
        for (let key = 0; key <= row; key++) {
          mixedValue += scores[key] * qkv[key * width * 3 + width * 2 + head * headDim + dim];
        }
        mixed[row * width + head * headDim + dim] = mixedValue;
      }
    }
  }
  const projected = linear(
    mixed,
    rows,
    width,
    tensor(`blocks.${layer}.attn.proj.weight`),
    tensor(`blocks.${layer}.attn.proj.bias`),
    width,
  );
  return { projected, last };
}

function mlp(x, rows, layer) {
  const hidden = nEmbd * 4;
  const up = linear(
    x,
    rows,
    nEmbd,
    tensor(`blocks.${layer}.mlp.0.weight`),
    tensor(`blocks.${layer}.mlp.0.bias`),
    hidden,
  );
  for (let index = 0; index < up.length; index++) up[index] = gelu(up[index]);
  return linear(
    up,
    rows,
    hidden,
    tensor(`blocks.${layer}.mlp.2.weight`),
    tensor(`blocks.${layer}.mlp.2.bias`),
    nEmbd,
  );
}

function forward(ids) {
  const rows = ids.length;
  const width = nEmbd;
  const hidden = new Float32Array(rows * width);
  const tokenWeight = tensor("tok_emb.weight");
  const positionWeight = tensor("pos_emb.weight");
  for (let row = 0; row < rows; row++) {
    const tokenOffset = ids[row] * width;
    const positionOffset = row * width;
    const hiddenOffset = row * width;
    for (let col = 0; col < width; col++) {
      hidden[hiddenOffset + col] = tokenWeight[tokenOffset + col] + positionWeight[positionOffset + col];
    }
  }
  const layers = [];
  for (let layer = 0; layer < nLayer; layer++) {
    const prefix = `blocks.${layer}.`;
    const attended = attention(layerNorm(hidden, rows, width, tensor(`${prefix}ln1.weight`), tensor(`${prefix}ln1.bias`)), rows, layer);
    for (let index = 0; index < hidden.length; index++) hidden[index] += attended.projected[index];
    const fed = mlp(layerNorm(hidden, rows, width, tensor(`${prefix}ln2.weight`), tensor(`${prefix}ln2.bias`)), rows, layer);
    for (let index = 0; index < hidden.length; index++) hidden[index] += fed[index];
    layers.push(attended.last);
  }
  const normed = layerNorm(hidden, rows, width, tensor("ln_f.weight"), tensor("ln_f.bias"));
  const last = normed.subarray((rows - 1) * width, rows * width);
  const logits = linear(last, 1, width, tensor("head.weight"), null, vocab);
  return { logits, layers };
}

function topK(logits) {
  const scored = [];
  let maxLogit = -Infinity;
  for (let index = 0; index < logits.length; index++) {
    if (logits[index] > maxLogit) maxLogit = logits[index];
  }
  let total = 0;
  const probs = new Float32Array(logits.length);
  for (let index = 0; index < logits.length; index++) {
    probs[index] = Math.exp(logits[index] - maxLogit);
    total += probs[index];
  }
  let best = 0;
  for (let index = 0; index < probs.length; index++) {
    probs[index] /= total;
    if (probs[index] > probs[best]) best = index;
  }
  for (let index = 0; index < probs.length; index++) scored.push({ index, p: probs[index] });
  scored.sort((a, b) => b.p - a.p);
  return {
    nextId: best,
    probs: scored.slice(0, TOP_K).map((item) => ({
      token: words[item.index] || "<unk>",
      p: Math.round(item.p * 10000) / 10000,
    })),
  };
}

function stats(promptLength, generated, windowLength) {
  return {
    parameters: paramCount,
    layers: nLayer,
    heads: nHead,
    embedding: nEmbd,
    context: blockSize,
    vocab,
    prompt: promptLength,
    generated,
    window: windowLength,
  };
}

function* think(userText, history) {
  const story = isStory(userText);
  const ids = promptIds(history, userText);
  const promptLength = ids.length;
  const endId = tokenToId.get("<end>");
  const userId = tokenToId.get("<user>");
  const replyIds = [];
  let sentenceEnds = 0;
  let hitLimit = true;
  const limit = story ? STORY_TOKENS : MAX_NEW_TOKENS;
  for (let stepIndex = 0; stepIndex < limit; stepIndex++) {
    const windowIds = ids.length > blockSize ? ids.slice(-blockSize) : ids.slice();
    const result = forward(windowIds);
    const choice = topK(result.logits);
    const label = words[choice.nextId] || "<unk>";
    const step = {
      type: "step",
      chosen: label,
      context: windowIds.map((id) => words[id] || "<unk>"),
      probs: choice.probs,
      layers: result.layers,
      reply: decode(replyIds),
    };
    const stop = choice.nextId === userId || choice.nextId === endId;
    if (stop) {
      step.stop = true;
      step.stats = stats(promptLength, replyIds.length, windowIds.length);
      yield step;
      hitLimit = false;
      break;
    }
    replyIds.push(choice.nextId);
    ids.push(choice.nextId);
    step.reply = decode(replyIds);
    step.stats = stats(promptLength, replyIds.length, windowIds.length);
    yield step;
    if (story && (label === "." || label === "?" || label === "!")) {
      sentenceEnds += 1;
      if (sentenceEnds >= STORY_SENTENCES) {
        hitLimit = false;
        break;
      }
    }
  }
  let text = decode(replyIds);
  if (hitLimit && !story) text = firstSentence(text);
  yield {
    type: "done",
    reply: text || FALLBACK,
    stats: stats(promptLength, replyIds.length, Math.min(promptLength + replyIds.length, blockSize)),
  };
}

function firstSentence(text) {
  const index = text.search(/[.!?]/);
  return index === -1 ? text.trim() : text.slice(0, index + 1).trim();
}

function installProfile(profile) {
  words = profile.words;
  tokenToId = new Map(words.map((word, index) => [word, index]));
  nLayer = profile.layers;
  nHead = profile.heads;
  nEmbd = profile.embedding;
  vocab = profile.vocabulary;
  blockSize = profile.context || 512;
  paramCount = profile.parameters;
}

function installSafetensors(buffer) {
  const view = new DataView(buffer);
  const headerLength = Number(view.getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, headerLength)));
  const dataStart = 8 + headerLength;
  tensors = new Map();
  Object.entries(header).forEach(([name, info]) => {
    if (name === "__metadata__" || info.dtype !== "F32") return;
    const start = dataStart + info.data_offsets[0];
    const end = dataStart + info.data_offsets[1];
    tensors.set(name, new Float32Array(buffer.slice(start, end)));
  });
}

async function ensureLoaded() {
  if (tensors.size) return;
  const profileUrl = new URL("model.json", self.location.href);
  const weightUrl = new URL("model.safetensors", self.location.href);
  const profile = await (await fetch(profileUrl)).json();
  installProfile(profile);
  const buffer = await (await fetch(weightUrl)).arrayBuffer();
  installSafetensors(buffer);
}

if (typeof WorkerGlobalScope !== "undefined" && self instanceof WorkerGlobalScope) {
  self.onmessage = async (event) => {
    if (event.data.type !== "ask") return;
    try {
      self.postMessage({ type: "status", text: "Loading the model…" });
      await ensureLoaded();
      self.postMessage({ type: "status", text: "Reading your line." });
      for (const item of think(event.data.message, event.data.history)) self.postMessage(item);
    } catch (error) {
      self.postMessage({ type: "error", message: error.message || "The model could not answer." });
    }
  };
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { installProfile, installSafetensors, think, decode };
}
