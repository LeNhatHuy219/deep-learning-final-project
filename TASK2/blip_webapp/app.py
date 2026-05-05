"""
Flask backend cho ứng dụng X-ray Captioning dùng BLIP fine-tuned.
Load checkpoint best_blip.pt (đã fine-tune trên Indiana University Chest X-rays)
và sinh báo cáo từ ảnh upload bằng beam search.
"""
import os
import re
import uuid
from pathlib import Path

import torch
from PIL import Image
from flask import Flask, render_template, request, jsonify, url_for
from deep_translator import GoogleTranslator
from transformers import BlipProcessor, BlipForConditionalGeneration


# ===== Cấu hình =====
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
CKPT_PATH = MODEL_DIR / "blip_final.pt"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {"png", "jpg", "jpeg"}
MAX_LEN = 128
NUM_BEAMS = 4
MODEL_NAME = "Salesforce/blip-image-captioning-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== Load model một lần khi khởi động =====
print(f"[init] Device: {DEVICE}")
print(f"[init] Loading BLIP processor & base model: {MODEL_NAME}")

processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)

# Load fine-tuned checkpoint nếu có
if CKPT_PATH.exists():
    print(f"[init] Loading fine-tuned checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", None)
    if val_loss is not None:
        print(f"[init] Đã load blip_final.pt (epoch {epoch}, val_loss={val_loss:.4f})")
    else:
        print(f"[init] Đã load blip_final.pt (epoch {epoch})")
    FINETUNED = True
else:
    print(
        f"[init] Không tìm thấy {CKPT_PATH} — đang dùng BLIP base (zero-shot). "
        f"Để có chất lượng tốt hơn, copy file blip_final.pt từ Google Drive vào folder model/."
    )
    FINETUNED = False

model.eval()
print("[init] Model sẵn sàng.")


# ===== Hàm clean placeholder XXXX của dataset Indiana =====
# Dataset Indiana University Chest X-rays đã ẩn danh thông tin nhạy cảm
# (tên, ngày, ID...) bằng token "XXXX" theo chuẩn HIPAA. Sau khi BLIP
# tokenizer (uncased) lowercase -> "xxxx". Model học theo và sinh ra
# "xxxx" trong output -> cần xoá khi hiển thị cho user.
def clean_xxxx(text: str) -> str:
    """Xoá các placeholder xxxx/XXXX và dọn dẹp khoảng trắng + dấu câu thừa."""
    if not isinstance(text, str):
        return text
    # 1) Xoá các chuỗi x lặp >= 2 lần (xx, xxx, xxxx, XXXX...)
    text = re.sub(r"\b[xX]{2,}\b", "", text)
    # 2) Dọn khoảng trắng thừa
    text = re.sub(r"\s+", " ", text)
    # 3) Dọn khoảng trắng trước dấu câu (vd "have  ." -> "have.")
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    # 4) Dọn dấu câu lặp (vd ".." -> ".", ", ," -> ",")
    text = re.sub(r"([.,;:!?])\s*\1+", r"\1", text)
    return text.strip()


# ===== Hàm sinh caption (beam search) =====
@torch.no_grad()
def generate_caption(image_path: str, max_length: int = MAX_LEN, num_beams: int = NUM_BEAMS) -> str:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    out = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    raw = processor.decode(out[0], skip_special_tokens=True)
    # Xoá xxxx ngay tại đây để mọi nơi gọi generate_caption() đều nhận output sạch
    return clean_xxxx(raw)


# ===== Helper =====
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def prettify(caption: str) -> str:
    """Viết hoa chữ đầu câu, thêm dấu chấm cuối nếu thiếu."""
    if not caption:
        return caption
    text = caption.strip()
    sentences = [s.strip() for s in text.replace(" . ", ". ").split(". ") if s.strip()]
    sentences = [s[0].upper() + s[1:] if s else s for s in sentences]
    text = ". ".join(sentences)
    if not text.endswith("."):
        text += "."
    return text


def translate_to_vi(text: str) -> str:
    """Dịch caption tiếng Anh sang tiếng Việt qua Google Translate."""
    if not text:
        return ""
    try:
        vi = GoogleTranslator(source="en", target="vi").translate(text)
        # Backup: nếu translator giữ nguyên xxxx trong bản dịch thì clean lại
        return clean_xxxx(vi)
    except Exception as e:
        print(f"[translate] lỗi: {e}")
        return ""


# ===== Flask app =====
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB


@app.route("/")
def index():
    return render_template("index.html", finetuned=FINETUNED)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Không có file ảnh nào được gửi lên."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Bạn chưa chọn file."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Định dạng không hợp lệ. Chỉ nhận PNG, JPG, JPEG."}), 400

    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = UPLOAD_DIR / unique_name
    file.save(save_path)

    try:
        caption = generate_caption(str(save_path))   # đã clean xxxx
        pretty = prettify(caption)
        caption_vi = translate_to_vi(pretty)         # dịch từ bản đã clean
    except Exception as e:
        return jsonify({"error": f"Lỗi khi sinh báo cáo: {e}"}), 500

    return jsonify({
        "caption": pretty,
        "caption_vi": caption_vi,
        "raw": caption,
        "image_url": url_for("static", filename=f"uploads/{unique_name}"),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)