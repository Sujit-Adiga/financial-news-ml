import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = "ProsusAI/finbert"


class FinBERTScorer:
    def __init__(self, batch_size=16):
        self.batch_size = batch_size
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME
        )
        self.model.to(self.device)
        self.model.eval()

        # ProsusAI/finbert uses:
        # 0 = positive, 1 = negative, 2 = neutral
        # Verify against the loaded model config rather than assuming it.
        self.id2label = {
            int(k): v.lower()
            for k, v in self.model.config.id2label.items()
        }

    def score(self, texts):
        rows = []

        for start in tqdm(
            range(0, len(texts), self.batch_size),
            desc="FinBERT"
        ):
            batch = texts[start:start + self.batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            with torch.no_grad():
                logits = self.model(**inputs).logits

            probs = torch.softmax(logits, dim=1).cpu()

            for p in probs:
                positive_idx = next(
                    i for i, label in self.id2label.items()
                    if "positive" in label
                )
                negative_idx = next(
                    i for i, label in self.id2label.items()
                    if "negative" in label
                )
                neutral_idx = next(
                    i for i, label in self.id2label.items()
                    if "neutral" in label
                )

                positive = float(p[positive_idx])
                negative = float(p[negative_idx])
                neutral = float(p[neutral_idx])

                rows.append({
                    "positive": positive,
                    "negative": negative,
                    "neutral": neutral,
                    "sentiment": positive - negative,
                    "label": (
                        "positive" if positive >= max(negative, neutral)
                        else "negative" if negative >= max(positive, neutral)
                        else "neutral"
                    ),
                })

        return pd.DataFrame(rows)
