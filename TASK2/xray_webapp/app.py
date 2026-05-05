"""
Flask backend cho ứng dụng X-ray Captioning.
Load model đã train (final_model_lstm.pth) và sinh báo cáo từ ảnh upload.
"""
import os
import uuid
from pathlib import Path
import json

import torch
from PIL import Image
from torchvision import transforms
from flask import Flask, render_template, request, jsonify, url_for
from deep_translator import GoogleTranslator

from model_def import EncoderCNN, DecoderLSTM, EMBED_SIZE, HIDDEN_SIZE


# ===== Cấu hình =====
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "final_model_lstm.pth"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {"png", "jpg", "jpeg"}
MAX_LEN = 80
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform giống lúc train
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ===== Load checkpoint một lần khi khởi động =====
print(f"[init] Device: {DEVICE}")
print(f"[init] Loading checkpoint từ: {MODEL_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy file model tại {MODEL_PATH}. "
        f"Hãy copy final_model_lstm.pth từ Google Drive vào folder model/."
    )

# weights_only=False vì checkpoint chứa cả vocab dict (không phải tensor thuần)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

vocab = checkpoint["vocab"]
idx2word = checkpoint["idx2word"]
vocab_size = len(vocab)

PAD_IDX = vocab["<pad>"]
SOS_IDX = vocab["<sos>"]
EOS_IDX = vocab["<eos>"]

# Một số checkpoint lưu idx2word với key dạng int, một số dạng str → chuẩn hoá về int
idx2word = {int(k): v for k, v in idx2word.items()}

encoder = EncoderCNN().to(DEVICE)
decoder = DecoderLSTM(
    vocab_size=vocab_size,
    embed_size=EMBED_SIZE,
    hidden_size=HIDDEN_SIZE,
    pad_idx=PAD_IDX,
).to(DEVICE)

encoder.load_state_dict(checkpoint["encoder"])
decoder.load_state_dict(checkpoint["decoder"])

encoder.eval()
decoder.eval()
print(f"[init] Model loaded — vocab size: {vocab_size}")


# ===== Hàm sinh caption (greedy) =====
def generate_caption(image_path: str, max_len: int = MAX_LEN) -> str:
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    result = []
    with torch.no_grad():
        encoder_out = encoder(image)

        h = torch.zeros(1, HIDDEN_SIZE, device=DEVICE)
        c = torch.zeros(1, HIDDEN_SIZE, device=DEVICE)

        word = torch.tensor([SOS_IDX], device=DEVICE)

        for _ in range(max_len):
            embedding = decoder.embedding(word)
            context, _ = decoder.attention(encoder_out, h)
            lstm_input = torch.cat([embedding, context], dim=1)
            h, c = decoder.lstm(lstm_input, (h, c))
            preds = decoder.fc(h)
            predicted = preds.argmax(1)
            word_idx = predicted.item()

            if word_idx == EOS_IDX:
                break

            result.append(idx2word.get(word_idx, "<unk>"))
            word = predicted

    return " ".join(result)


# ===== Helper =====
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def prettify(caption: str) -> str:
    """Viết hoa chữ đầu câu, thêm dấu chấm cuối nếu thiếu."""
    if not caption:
        return caption
    sentences = [s.strip() for s in caption.split(" . ") if s.strip()]
    sentences = [s[0].upper() + s[1:] if s else s for s in sentences]
    text = ". ".join(sentences)
    if not text.endswith("."):
        text += "."
    return text


def translate_to_vi(text: str) -> str:
    """Dịch caption tiếng Anh sang tiếng Việt qua Google Translate.
    Không raise — nếu fail thì trả về chuỗi rỗng để FE biết bỏ qua."""
    if not text:
        return ""
    try:
        return GoogleTranslator(source="en", target="vi").translate(text)
    except Exception as e:
        print(f"[translate] lỗi: {e}")
        return ""


# ===== Flask app =====
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Không có file ảnh nào được gửi lên."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Bạn chưa chọn file."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Định dạng không hợp lệ. Chỉ nhận PNG, JPG, JPEG."}), 400

    # Lưu file với tên random để tránh trùng
    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = UPLOAD_DIR / unique_name
    file.save(save_path)

    try:
        caption = generate_caption(str(save_path))
        pretty = prettify(caption)
        caption_vi = translate_to_vi(pretty)
    except Exception as e:
        return jsonify({"error": f"Lỗi khi sinh báo cáo: {e}"}), 500

    return jsonify({
        "caption": pretty,
        "caption_vi": caption_vi,
        "raw": caption,
        "image_url": url_for("static", filename=f"uploads/{unique_name}"),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)