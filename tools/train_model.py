#!/usr/bin/env python3
"""
Train FastText binary classifier.
Run: python tools/train_model.py --data data/training.txt --out model.ftz
Your i3 10th gen will finish this in ~2-3 minutes.
"""
import argparse
import re
import fasttext
import random
from pathlib import Path

# NumPy 2.0+ compatibility fix for FastText
try:
    import numpy as np
    _orig_array = np.array
    def _fixed_array(obj, *args, **kwargs):
        if kwargs.get('copy') is False:
            kwargs.pop('copy')
            return np.asarray(obj, *args, **kwargs)
        return _orig_array(obj, *args, **kwargs)
    np.array = _fixed_array
except ImportError:
    pass


def split_train_test(data_path: str, test_ratio: float = 0.1) -> tuple[str, str]:
    """Split data into train/test files."""
    lines = Path(data_path).read_text(encoding="utf-8").strip().splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - test_ratio))
    train_path = data_path.replace(".txt", "_train.txt")
    test_path = data_path.replace(".txt", "_test.txt")
    Path(train_path).write_text("\n".join(lines[:split]), encoding="utf-8")
    Path(test_path).write_text("\n".join(lines[split:]), encoding="utf-8")
    return train_path, test_path


def train(data_path: str, out_path: str) -> None:
    train_path, test_path = split_train_test(data_path)

    print("🧠 Training FastText...")
    # Parameters optimized for Indonesian slang
    model = fasttext.train_supervised(
        train_path,
        epoch=30,
        lr=0.5,
        wordNgrams=2,   # bigrams handle "gak ada" vs "ada" distinction
        dim=50,         # smaller = less RAM (50 vs default 100)
        loss="softmax",
        minn=3,         # subword min — catches "sopi"/"sfood"/"sopie"
        maxn=6,         # subword max
        thread=4,       # utilization of multiple cores
    )

    # Evaluate before quantizing
    result = model.test(test_path)
    n, precision, recall = result
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"📊 Test set: {n} samples | Precision: {precision:.2%} | Recall: {recall:.2%}")
    print(f"   F1 Score: {f1:.2%}")

    print("📉 Quantizing and compressing...")
    # Quantize → compress to ~10-20MB for VPS deployment
    model.quantize(train_path, retrain=True, qnorm=True, cutoff=100_000)
    model.save_model(out_path)

    size_mb = Path(out_path).stat().st_size / 1_048_576
    print(f"✅ Saved compressed model → {out_path} ({size_mb:.1f}MB)")

    # Quick sanity check
    print("\n🔍 Sanity checks:")
    test_cases = [
        "gercep gais shopee food cb 70rb", # Promo
        "jalanan sini macet gak?",         # Junk
        "aman jsm alfa luber bgt",         # Promo
        "tanya dong ojol daerah sini",     # Junk
        "voc grabfood cair gais",          # Promo
    ]
    for t in test_cases:
        # FastText predict expects a string and returns a tuple of (labels, probabilities)
        # We wrap in try/except to catch NumPy 2.0 incompatibilities if they persist
        try:
            labels, probs = model.predict(t.lower(), k=1)
            label = labels[0]
            prob = probs[0]
            print(f"  [{label:18s} {prob:.0%}] {t[:55]}")
        except Exception as e:
            print(f"  [ERROR             ] {t[:55]} -> {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="model.ftz")
    args = p.parse_args()
    train(args.data, args.out)
