# Chest X-ray Report Generator — Flask Web App

Web demo cho model DenseNet121 + LSTM Attention sinh báo cáo X-quang ngực.

## Cấu trúc thư mục

```
xray_webapp/
├── app.py                    # Flask backend
├── model_def.py              # Định nghĩa kiến trúc model
├── requirements.txt          # Các thư viện cần cài
├── model/
│   └── final_model_lstm.pth   # ← copy từ Google Drive vào đây
├── templates/
│   └── index.html            # Giao diện
└── static/
    ├── css/style.css
    └── uploads/              # Folder lưu ảnh user upload (tự tạo)
```

## Các bước chạy

### 1. Tạo môi trường ảo

```bash
cd xray_webapp
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Cài thư viện

```bash
pip install -r requirements.txt
```

Nếu máy có GPU NVIDIA, cài torch bản CUDA cho nhanh hơn (lệnh thay torch ở `requirements.txt`):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Copy model từ Google Drive

Vào Google Drive → folder `xray_captioning` → tải file `final_model_lstm.pth` về và bỏ vào folder `model/` của project.

### 4. Chạy server

```bash
python app.py
```

Truy cập: http://localhost:5000

Lần đầu khởi động sẽ tải DenseNet121 weights từ torchvision (~30MB) và load checkpoint, mất ~10–30 giây.

## Lưu ý

- File model phải đúng tên `final_model_lstm.pth` và đặt trong `model/`. Nếu muốn đổi, sửa biến `MODEL_PATH` trong `app.py`.
- Ảnh upload được lưu tạm vào `static/uploads/`. Có thể clear định kỳ.
- Decode đang dùng greedy (chọn từ có xác suất cao nhất ở mỗi bước). Muốn chất lượng cao hơn → đổi sang beam search.

## Deploy production

App đang chạy bằng dev server của Flask, chỉ phù hợp test local. Khi deploy thật:

```bash
pip install gunicorn
gunicorn -w 1 -b 0.0.0.0:5000 --timeout 120 app:app
```

Lưu ý `-w 1` (1 worker) — vì model nặng, mỗi worker load 1 bản model riêng tốn RAM.
