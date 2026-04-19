---
language:
- ur
license: mit
task_categories:
- text-to-speech
tags:
- urdu
- tts
- latents
- dacvae
- audio
- parquet
size_categories:
- 10K<n<100K
---

# 🎙️ Urdu TTS Latent Dataset — munch-1-latent-NEW-parquet

Pre-computed DACVAE latent representations for **51,021 Urdu utterances**, ready for TTS model training. No audio decoding required at training time — load the dataset, reshape the binary blob, and train.

---

## Source

| Field | Value |
|---|---|
| Source audio | [`Humair332/Urdu-munch-1`](https://huggingface.co/datasets/Humair332/Urdu-munch-1) |
| Codec | [`Aratako/Semantic-DACVAE-Japanese-32dim`](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim) |
| Codec sample rate | 48,000 Hz |
| Encoder hop size | 1,920 samples |
| Latent frame rate | **25.0 Hz** |
| Latent dim (D) | **32** |
| Duration formula | `num_frames × 1920 / 48000` = `num_frames / 25.0` |

---

## Dataset Stats

| Stat | Value |
|---|---|
| Total rows | 51,021 |
| Total file size | 2.36 GB |
| Duration range | 0.56 s — 45 s |
| Frames range | 14 — 1,130 |
| Speakers | 13 voices |
| Language | Urdu (اردو) |

### Speakers

`alloy` · `echo` · `fable` · `nova` · `shimmer` · `verse` · `ballad` · `ash` · `sage` · `amuch` · `onyx` · `coral` · `openai`

---

## Schema

| Column | Type | Description |
|---|---|---|
| `text` | `string` | Normalised Urdu transcript |
| `latent` | `binary` | Raw float32 bytes — shape `(num_frames, 32)`, row-major |
| `latent_dim` | `int32` | Always `32` — needed to reshape the binary blob |
| `num_frames` | `int32` | Number of latent frames (T) |
| `duration` | `float32` | Audio duration in seconds = `num_frames / 25.0` |
| `speaker_id` | `string` | Speaker label, format `urdu-munch-1:<voice>` |

> **Why binary?** Storing latents as raw `float32` bytes instead of nested lists or JSON reduces file size by ~7× (384 MB → 54 MB per batch) with zero precision loss and O(1) reshape on load.

---

## Load & Use

### Basic load

```python
import numpy as np
import torch
from datasets import load_dataset

ds = load_dataset("zuhri025/munch-1-latent-NEW-parquet", split="train")

row = ds[0]
print(row["text"])       # اسلام آباد عالمی بینک...
print(row["duration"])   # 14.36

# reshape binary -> (T, D) float32
latent = np.frombuffer(row["latent"], dtype=np.float32).reshape(
    row["num_frames"], row["latent_dim"]
)
print(latent.shape)      # (359, 32)

# as torch tensor
latent_t = torch.from_numpy(latent.copy())   # .copy() required — frombuffer is read-only
```

### PyTorch DataLoader with padding

```python
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

ds = load_dataset("zuhri025/munch-1-latent-NEW-parquet", split="train")

def collate_fn(batch):
    latents = [
        torch.from_numpy(
            np.frombuffer(b["latent"], dtype=np.float32)
              .reshape(b["num_frames"], b["latent_dim"])
              .copy()
        )
        for b in batch
    ]
    max_T = max(t.shape[0] for t in latents)
    D     = latents[0].shape[1]                      # 32

    padded = torch.zeros(len(latents), max_T, D)
    mask   = torch.zeros(len(latents), max_T, dtype=torch.bool)
    for i, t in enumerate(latents):
        padded[i, :t.shape[0]] = t
        mask[i,   :t.shape[0]] = True

    return {
        "text":       [b["text"]       for b in batch],
        "speaker_id": [b["speaker_id"] for b in batch],
        "latent":     padded,                        # (B, T_max, 32)
        "mask":       mask,                          # (B, T_max)
        "duration":   torch.tensor([b["duration"] for b in batch]),
    }

loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_fn)

for batch in loader:
    print(batch["latent"].shape)   # torch.Size([8, T_max, 32])
    print(batch["mask"].shape)     # torch.Size([8, T_max])
    break
```

### Filter by speaker or duration

```python
# single speaker
ds_alloy = ds.filter(lambda x: x["speaker_id"] == "urdu-munch-1:alloy")

# utterances under 10 seconds
ds_short = ds.filter(lambda x: x["duration"] < 10.0)

# utterances between 3 and 20 seconds (typical TTS training range)
ds_clean = ds.filter(lambda x: 3.0 <= x["duration"] <= 20.0)
```

### Decode latent back to audio

```python
import torch
import soundfile as sf
from huggingface_hub import hf_hub_download
import sys

# Load codec
try:
    from dacvae import DACVAE
except ImportError:
    #pip install git+https://github.com/facebookresearch/dacvae
    raise

weights = hf_hub_download("Aratako/Semantic-DACVAE-Japanese-32dim", "weights.pth")
model = DACVAE.load(weights).eval()

row = ds[0]
latent = torch.from_numpy(
    np.frombuffer(row["latent"], dtype=np.float32)
      .reshape(row["num_frames"], row["latent_dim"])
      .copy()
).unsqueeze(0)                    # (1, T, 32)

with torch.inference_mode():
    audio = model.decode(latent.transpose(1, 2))   # (1, 1, samples)

audio_np = audio.squeeze().numpy()
sf.write("output.wav", audio_np, 48000)
```

---

## Pipeline

```
Humair332/Urdu-munch-1          ← raw audio (22,050 Hz) + Urdu transcripts
        ↓  precompute_urdu_latents.py
zuhri025/munch-1-latent-NEW     ← JSONL with latents as float32 lists
        ↓  jsonl_to_parquet.py
zuhri025/munch-1-latent-NEW-parquet  ← this dataset (binary parquet, 25 Hz)
```

---

## Citation

If you use this dataset, please also credit the source audio dataset and codec:

- Source audio: `Humair332/Urdu-munch-1`
- Codec: `Aratako/Semantic-DACVAE-Japanese-32dim` — [Irodori-TTS](https://github.com/Aratako/Irodori-TTS)