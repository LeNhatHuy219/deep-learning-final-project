"""
evaluate_test.py — Tính 'test accuracy' cho model BLIP fine-tuned.

LƯU Ý: Đây là task captioning (sinh báo cáo X-quang) nên 'accuracy' không
phải là 1 con số duy nhất. Script tính bộ metric chuẩn:
    - BLEU-1, BLEU-4
    - ROUGE-L
    - METEOR
    - BERTScore (F1)
    - Exact Match (% caption khớp 100% sau khi clean) — chỉ để tham khảo

Script KHÔNG sửa file nào có sẵn (app.py, notebook). Chạy độc lập:

    cd blip_webapp
    source venv/bin/activate
    pip install rouge-score bert-score nltk sacrebleu kagglehub scikit-learn pandas
    python evaluate_test.py

Trùng split với notebook (random_state=42, test_size=0.2 -> 0.5) nên test set
giống hệt lúc train.
"""
import os
import re
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from transformers import BlipProcessor, BlipForConditionalGeneration


# ===== Config (giống notebook) =====
BASE_DIR    = Path(__file__).resolve().parent
MODEL_DIR   = BASE_DIR / "model"
DEFAULT_CKPT = MODEL_DIR / "best_blip.pt"
MODEL_NAME  = "Salesforce/blip-image-captioning-base"
MAX_LEN     = 128
NUM_BEAMS   = 4
GEN_BATCH   = 8
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clean_xxxx(text: str) -> str:
    """Xoá placeholder XXXX (HIPAA) — giống logic trong app.py & notebook."""
    if not isinstance(text, str):
        return text
    text = re.sub(r"\b[xX]{2,}\b", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"([.,;:!?])\s*\1+", r"\1", text)
    return text.strip()


def load_dataset(data_dir: Path) -> pd.DataFrame:
    """Đọc CSV, map uid -> ảnh, giống cell 10 + 12 của notebook."""
    csv_path = data_dir / "indiana_reports.csv"
    img_dir  = data_dir / "images"
    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {csv_path}")
    if not img_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy folder ảnh {img_dir}")

    df = pd.read_csv(csv_path)
    df = df[["uid", "findings"]].dropna(subset=["findings"])
    df = df[df["findings"].str.strip() != ""].reset_index(drop=True)

    # Map uid -> path ảnh đầu tiên
    from collections import defaultdict
    uid_to_paths = defaultdict(list)
    for root, _, files in os.walk(img_dir):
        for f in files:
            if f.endswith(".png") and "_" in f:
                uid_to_paths[f.split("_")[0]].append(os.path.join(root, f))

    df["image_path"] = df["uid"].apply(
        lambda u: uid_to_paths[str(u)][0] if uid_to_paths.get(str(u)) else None
    )
    df = df.dropna(subset=["image_path"]).reset_index(drop=True)
    return df


def make_test_split(df: pd.DataFrame) -> pd.DataFrame:
    """Recreate đúng split của notebook (random_state=42)."""
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df,  test_df  = train_test_split(temp_df, test_size=0.5, random_state=42)
    return test_df.reset_index(drop=True)


@torch.no_grad()
def generate_batch(model, processor, image_paths, num_beams=NUM_BEAMS, max_length=MAX_LEN):
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = processor(images=images, return_tensors="pt").to(DEVICE)
    out = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    return [processor.decode(o, skip_special_tokens=True) for o in out]


def compute_metrics(refs_text, preds_text):
    """Tính BLEU/ROUGE-L/METEOR/BERTScore + Exact-Match."""
    import nltk
    for r in ("wordnet", "omw-1.4", "punkt", "punkt_tab"):
        nltk.download(r, quiet=True)
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer

    refs_tok  = [[r.lower().split()] for r in refs_text]
    preds_tok = [p.lower().split()  for p in preds_text]

    smooth = SmoothingFunction().method1
    bleu1 = corpus_bleu(refs_tok, preds_tok, weights=(1, 0, 0, 0),         smoothing_function=smooth)
    bleu4 = corpus_bleu(refs_tok, preds_tok, weights=(.25,.25,.25,.25),    smoothing_function=smooth)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = float(np.mean([scorer.score(r, p)["rougeL"].fmeasure
                             for r, p in zip(refs_text, preds_text)]))

    meteor_avg = float(np.mean([meteor_score([r[0]], p)
                                for r, p in zip(refs_tok, preds_tok)]))

    # BERTScore (cần torch + transformers — đã có sẵn)
    from bert_score import score as bertscore
    _, _, F1 = bertscore(preds_text, refs_text, lang="en", verbose=False)
    bert_f1 = float(F1.mean().item())

    # Exact match (chỉ để tham khảo, thường rất thấp với captioning)
    em = float(np.mean([1.0 if r.strip().lower() == p.strip().lower() else 0.0
                        for r, p in zip(refs_text, preds_text)]))

    return {
        "BLEU-1":      bleu1,
        "BLEU-4":      bleu4,
        "ROUGE-L":     rouge_l,
        "METEOR":      meteor_avg,
        "BERTScore":   bert_f1,
        "ExactMatch":  em,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",     default=str(DEFAULT_CKPT),
                    help="Đường dẫn checkpoint (.pt). Default: model/best_blip.pt")
    ap.add_argument("--data_dir", default=None,
                    help="Folder dataset Indiana (chứa indiana_reports.csv + images/). "
                         "Nếu bỏ trống sẽ tự tải bằng kagglehub.")
    ap.add_argument("--limit",    type=int, default=None,
                    help="Chỉ chạy trên N sample đầu của test set (debug nhanh).")
    ap.add_argument("--out",      default=str(BASE_DIR / "test_predictions.json"))
    args = ap.parse_args()

    print(f"[init] Device: {DEVICE}")

    # ===== Dataset =====
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        print("[data] Tải dataset bằng kagglehub (cần kaggle.json đã setup)…")
        import kagglehub
        data_dir = Path(kagglehub.dataset_download("raddar/chest-xrays-indiana-university"))
    print(f"[data] DATA_DIR = {data_dir}")

    df      = load_dataset(data_dir)
    test_df = make_test_split(df)
    if args.limit:
        test_df = test_df.head(args.limit).reset_index(drop=True)
    print(f"[data] Test samples: {len(test_df)}")

    # ===== Load model =====
    print(f"[model] Loading {MODEL_NAME} + checkpoint {args.ckpt}")
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model     = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)

    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"[model] Loaded (epoch={ckpt.get('epoch','?')}, "
          f"val_loss={ckpt.get('val_loss','?')})")

    # ===== Generate predictions =====
    predictions = []
    for i in tqdm(range(0, len(test_df), GEN_BATCH), desc="Generating"):
        paths = test_df.iloc[i:i+GEN_BATCH]["image_path"].tolist()
        predictions.extend(generate_batch(model, processor, paths))

    # ===== Clean xxxx ở cả ref và pred trước khi tính metric =====
    refs_clean  = [clean_xxxx(r) for r in test_df["findings"].tolist()]
    preds_clean = [clean_xxxx(p) for p in predictions]
    keep = [i for i, (r, p) in enumerate(zip(refs_clean, preds_clean))
            if r.strip() and p.strip()]
    refs_eval  = [refs_clean[i]  for i in keep]
    preds_eval = [preds_clean[i] for i in keep]
    print(f"[eval] Số cặp dùng để eval: {len(refs_eval)} / {len(predictions)}")

    # ===== Lưu prediction để khỏi sinh lại =====
    with open(args.out, "w") as f:
        json.dump({
            "refs_raw":  test_df["findings"].tolist(),
            "preds_raw": predictions,
            "refs":      refs_clean,
            "preds":     preds_clean,
        }, f, ensure_ascii=False)
    print(f"[eval] Đã lưu predictions vào {args.out}")

    # ===== Metrics =====
    print("[eval] Đang tính metric (BERTScore lần đầu sẽ tải ~400MB)…")
    metrics = compute_metrics(refs_eval, preds_eval)

    print("\n========== TEST METRICS ==========")
    for k, v in metrics.items():
        print(f"{k:<11s}: {v:.4f}")
    print("==================================")
    print("Lưu ý: caption-task không có 'accuracy' đơn lẻ. ExactMatch chỉ để tham khảo;")
    print("BLEU-4 / ROUGE-L / METEOR / BERTScore mới là chỉ số chính.")

    # Lưu metric vào file
    with open(BASE_DIR / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[eval] Đã lưu metric vào {BASE_DIR / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
