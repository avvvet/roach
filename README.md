# speech-to-text (STT) Fine-tuning Guide

> Ruach (רוּחַ — pronounced "ROO-akh" — breath, spirit, voice) by Awet Tsegazeab

a fine-tuned speech-to-text model
built on top of OpenAI's Whisper. This guide walks through everything from dataset
preparation to deploying the final model on your machine for any low resource language.

---

## What Is This?

We take Whisper — a powerful multilingual speech recognition model by OpenAI —
and teach it to accurately recognize a target low-resource language.

The process is called **fine-tuning**: we show the model thousands of audio clips
paired with their correct transcriptions, and it gradually adjusts itself to get
better at that language specifically.
```
Any language + Any dataset → same pipeline → working STT model
```


## Folder Structure

```
ruach/
├── finetune/
│   ├── train.py              ← training script
│   ├── docker-compose.yml    ← docker config
│   ├── Dockerfile            ← docker image definition
│   └── output/               ← trained model saved here
│
├── dataset/
│   ├── data/
│   │   ├── train/
│   │   │   ├── metadata.csv  ← transcriptions for train clips
│   │   │   └── audio/        ← 9,340 WAV files
│   │   └── test/
│   │       ├── metadata.csv  ← transcriptions for test clips
│   │       └── audio/        ← 300 WAV files
│   └── cache/                ← HuggingFace feature cache (auto-created)
│
└── models/
    └── whisper-lang-ct2/  ← final converted model (after training)
```


**Model options:**
```
openai/whisper-small    → 244M params, faster, lower accuracy  (~70% WER)
openai/whisper-medium   → 769M params, slower, better accuracy (~40-50% WER)
openai/whisper-large-v3 → 1.5B params, slowest, best accuracy  (~25-35% WER)
```

Start with `whisper-medium` for the best balance of speed and accuracy.


## Step 7 — Reading the Training Output

You will see two types of lines:

**Training loss** (every 5 steps):
```
{'loss': 0.45, 'grad_norm': 12.3, 'learning_rate': 9.9e-06, 'epoch': 0.14}
```
- `loss` → how wrong the model is. Should drop from ~4.0 toward ~0.05
- `epoch` → progress through the dataset (1.0 = seen all clips once)

**Evaluation WER** (end of each epoch):
```
{'eval_loss': 0.25, 'eval_wer': 55.3, 'eval_runtime': 880, 'epoch': 1.0}
```
- `eval_wer` → Word Error Rate on test set. Lower is better.
- At epoch 1: expect ~55-65% for whisper-medium
- At epoch 5: expect ~40-50%
- Early stopping triggers if no improvement for 3 epochs

**Healthy training looks like:**
```
Epoch 1  loss: 0.8   WER: 60%
Epoch 2  loss: 0.3   WER: 52%
Epoch 3  loss: 0.1   WER: 46%
Epoch 4  loss: 0.05  WER: 43%
Epoch 5  loss: 0.03  WER: 43%   ← no improvement
Epoch 6  loss: 0.02  WER: 44%   ← worse
Epoch 7  loss: 0.01  WER: 44%   ← 3 epochs no improvement → STOP
→ Best model (epoch 4, WER 43%) saved automatically
```

## Pipeline in Action

![Ruach fine-tuning pipeline running on RTX 3090](ruach_screenshot.png)



## Understanding WER (Word Error Rate)

WER measures how many words the model gets wrong:

```
100 words spoken        WER 0%   → perfect
100 words spoken        WER 30%  → 30 words wrong, 70 correct
100 words spoken        WER 70%  → 70 words wrong, 30 correct
```

*Ruach (רוּחַ) — "ROO-akh" — by Awet Tsegazeab, Amharic STT model*