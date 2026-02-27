"""
Whisper Fine-tuning for Amharic (am)
Based on: https://huggingface.co/blog/fine-tune-whisper
Adapted for local dataset with metadata.csv format

Changes from whisper-small run:
  - BASE_MODEL default → whisper-medium
  - per_device_train_batch_size → 4 (medium fits on 3090 with batch 4)
  - gradient_accumulation_steps → 2 (effective batch = 8, same as before)
  - early_stopping_patience → 3 (stop if no WER improvement for 3 epochs)
  - save_total_limit → 3 (keep only top 3 checkpoints, saves disk)
  - dataloader_num_workers → 4 (faster CPU data loading)
  - warmup_steps → 500 (medium needs longer warmup)
  - fp16 → True (medium fits in fp16 on 3090)
"""

import os
import csv
import torch
import evaluate
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import soundfile as sf
from datasets import Dataset, DatasetDict, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)

# ─── Config from environment ───────────────────────────────────────────────────
os.environ["HF_DATASETS_CACHE"] = "/app/dataset/cache"
BASE_MODEL  = os.getenv("BASE_MODEL",  "openai/whisper-medium")
LANGUAGE    = os.getenv("LANGUAGE",    "amharic")
EPOCHS      = int(os.getenv("EPOCHS",  "20"))
BATCH_SIZE  = int(os.getenv("BATCH_SIZE", "4"))
DATASET_DIR = os.getenv("DATASET_DIR", "/app/dataset/data")
OUTPUT_DIR  = os.getenv("OUTPUT_DIR",  "/app/output")

print(f"Base model : {BASE_MODEL}")
print(f"Language   : {LANGUAGE}")
print(f"Epochs     : {EPOCHS}")
print(f"Batch size : {BATCH_SIZE}")
print(f"Dataset    : {DATASET_DIR}")
print(f"Output     : {OUTPUT_DIR}")
print(f"GPU        : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ─── Load dataset from metadata.csv ────────────────────────────────────────────
def load_split(split_dir):
    """Load audio + transcription from metadata.csv"""
    metadata_path = os.path.join(split_dir, "metadata.csv")
    rows = []
    with open(metadata_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_path = os.path.join(split_dir, row["file_name"])
            rows.append({
                "audio": audio_path,
                "sentence": row["transcription"],
            })
    return Dataset.from_list(rows)

print("\nLoading dataset...")
dataset = DatasetDict({
    "train": load_split(os.path.join(DATASET_DIR, "train")),
    "test":  load_split(os.path.join(DATASET_DIR, "test")),
})
print(f"Train: {len(dataset['train'])} samples")
print(f"Test : {len(dataset['test'])} samples")

# Cast audio column — loads WAV and resamples to 16kHz automatically
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# ─── Processor ─────────────────────────────────────────────────────────────────
print(f"\nLoading processor for {BASE_MODEL}...")
feature_extractor = WhisperFeatureExtractor.from_pretrained(BASE_MODEL)
tokenizer = WhisperTokenizer.from_pretrained(BASE_MODEL, language=LANGUAGE, task="transcribe")
processor = WhisperProcessor.from_pretrained(BASE_MODEL, language=LANGUAGE, task="transcribe")

# ─── Prepare features ──────────────────────────────────────────────────────────
def prepare_dataset(batch):
    import soundfile as sf
    audio = batch["audio"]
    array, sampling_rate = sf.read(audio["path"])
    batch["input_features"] = feature_extractor(
        array, sampling_rate=sampling_rate
    ).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

print("Preparing features...")
dataset = dataset.map(
    prepare_dataset,
    remove_columns=dataset.column_names["train"],
    num_proc=1,
    writer_batch_size=50,
    keep_in_memory=False,
    cache_file_names={
        "train": "/app/dataset/cache/train_features.arrow",
        "test": "/app/dataset/cache/test_features.arrow",
    }
)

# ─── Data collator ─────────────────────────────────────────────────────────────
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# ─── Metrics ───────────────────────────────────────────────────────────────────
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# ─── Model ─────────────────────────────────────────────────────────────────────
print(f"\nLoading model {BASE_MODEL}...")
model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.generation_config.language = LANGUAGE
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

# ─── Training arguments ────────────────────────────────────────────────────────
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,

    # Batch — tuned for RTX 3090 24GB with whisper-medium
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,       # effective batch = BATCH_SIZE * 2 = 8

    # Learning rate — medium needs more warmup than small
    learning_rate=1e-5,
    warmup_steps=500,

    # Memory
    fp16=True,                           # whisper-medium fits in fp16 on 3090
    optim="adafactor",

    # Speed — more workers feed GPU faster
    dataloader_num_workers=4,

    # Evaluation
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=5,

    # Generation
    predict_with_generate=True,
    generation_max_length=225,

    # Best model + early stopping
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    save_total_limit=3,                  # keep only 3 best checkpoints

    report_to="none",                    # no wandb/tensorboard needed
)

# ─── Trainer ───────────────────────────────────────────────────────────────────
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3)  # stop if WER doesn't improve for 3 epochs
    ],
)

# ─── Train ─────────────────────────────────────────────────────────────────────
print("\nStarting fine-tuning...")
trainer.train()

# ─── Save ──────────────────────────────────────────────────────────────────────
print(f"\nSaving model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print("Done.")