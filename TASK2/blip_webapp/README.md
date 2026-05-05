# Chest X-ray Report Generator (BLIP) — Flask Web App

Web demo cho model **BLIP fine-tuned** (`Salesforce/blip-image-captioning-base`)
sinh báo cáo X-quang ngực — train trong notebook `cau2_blip_xray.ipynb`.

## Cấu trúc thư mục

```
blip_webapp/
├── app.py                # Flask backend (load BLIP + checkpoint)
├── requirements.txt
├── model/
│   └── best_blip.pt      # ← copy từ Google Drive vào đây
├── templates/
│   └── index.html
└── static/
    ├── css/style.css
    └── uploads/          # ảnh user upload (tự tạo)
```

## Các bước chạy

### 1. Tạo môi trường ảo

```bash
cd blip_webapp
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 2. Cài thư viện

```bash
pip install -r requirements.txt
```

Có GPU NVIDIA thì cài torch CUDA cho nhanh:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Copy checkpoint từ Google Drive

Trong notebook, model được lưu tại `/content/drive/MyDrive/xray_blip/best_blip.pt`.
Tải file đó về và bỏ vào folder `model/` của project (đúng tên `best_blip.pt`).

> Nếu chưa có checkpoint, app vẫn chạy được — sẽ tự fallback sang BLIP base
> (zero-shot). Output sẽ rất chung chung vì BLIP base train trên ảnh thường,
> không phải ảnh y khoa.

### 4. Chạy server

```bash
python app.py
```

Truy cập: <http://localhost:5002>

Lần đầu khởi động sẽ tải BLIP weights từ HuggingFace (~990MB) và load checkpoint
fine-tune, mất ~30–60 giây.

## Lưu ý

- Checkpoint phải đúng định dạng `{"model_state": ..., "epoch": ..., "val_loss": ...}` — đúng cách lưu trong notebook.
- File phải đúng tên `best_blip.pt` đặt trong `model/`. Đổi tên thì sửa `CKPT_PATH` trong `app.py`.
- Decode dùng beam search (`num_beams=4`, `no_repeat_ngram_size=3`, `max_length=128`) — giống lúc đánh giá trong notebook.
- Ảnh upload lưu tạm vào `static/uploads/`. Nên clear định kỳ.
- App chạy port **5002** (xray_webapp gốc dùng 5001 nên có thể chạy song song).

## Deploy production

Dev server của Flask chỉ phù hợp test local. Khi deploy thật:

```bash
pip install gunicorn
gunicorn -w 1 -b 0.0.0.0:5002 --timeout 180 app:app
```

`-w 1` (1 worker) — BLIP base ~250M params, mỗi worker load 1 bản tốn RAM/VRAM.
