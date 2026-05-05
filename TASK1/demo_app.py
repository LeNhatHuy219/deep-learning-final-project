"""
demo_app.py - Demo VQA Tieng Viet - UI Premium
python demo_app.py -> http://localhost:7860
"""
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import os
import gc
import json
import math
import torch
import torch.nn as nn
import unicodedata
from PIL import Image
from collections import Counter, OrderedDict
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoModel, AutoTokenizer, BlipProcessor, BlipForQuestionAnswering
from underthesea import word_tokenize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, 'vqa_dataset_final')
TRAIN_JSON = os.path.join(BASE_DIR, 'vqa_train.json')
FALLBACK_JSON = os.path.join(BASE_DIR, 'vqa_dataset_flattened.json')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ── Text preprocessing ────────────────────────────────────────
MAX_Q_LEN, MAX_A_LEN = 40, 30
def segment_vietnamese(text): return word_tokenize(text, format="text")
phobert_tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
def encode_text(text, max_len=MAX_Q_LEN):
    return phobert_tokenizer(segment_vietnamese(text), max_length=max_len,
                             padding='max_length', truncation=True, return_tensors='pt')

# ── Answer vocab ──────────────────────────────────────────────
SPECIAL_TOKENS = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
train_json_path = TRAIN_JSON if os.path.exists(TRAIN_JSON) else FALLBACK_JSON
if not os.path.exists(train_json_path):
    raise FileNotFoundError(
        'Missing VQA data file. Put vqa_train.json or vqa_dataset_flattened.json in TASK1/.'
    )
with open(train_json_path, 'r', encoding='utf-8') as f:
    train_data = json.load(f)
word_counter = Counter()
for s in train_data:
    word_counter.update(segment_vietnamese(s['answer']).lower().split())
vocab = OrderedDict()
for tok in SPECIAL_TOKENS: vocab[tok] = len(vocab)
for w, freq in word_counter.most_common():
    if freq >= 1: vocab[w] = len(vocab)
idx_to_word = {i: w for w, i in vocab.items()}
VOCAB_SIZE = len(vocab)
answer_counter = Counter(s['answer'].lower().strip() for s in train_data)
ans_to_idx = {a: i for i, (a, _) in enumerate(answer_counter.most_common())}
idx_to_ans = {i: a for a, i in ans_to_idx.items()}
NUM_ANSWERS = len(ans_to_idx)

def decode_answer_seq(ids):
    words = []
    for i in ids:
        w = idx_to_word.get(i, '<UNK>')
        if w == '<EOS>': break
        if w not in ('<PAD>', '<BOS>'): words.append(w)
    return ' '.join(words)

# ── Diacritics restoration dictionary ─────────────────────────
def strip_accents(text):
    t = unicodedata.normalize('NFD', text.lower().strip())
    return ''.join(c for c in t if unicodedata.category(c) != 'Mn')

# Count how often each raw word (no accent) appears in training data AS-IS
_plain_word_freq = Counter()
for s in train_data:
    for word in s['answer'].lower().split():
        if strip_accents(word) == word:          # word itself has no diacritics
            _plain_word_freq[word] += 1

diacritics_map = {}
for s in train_data:
    for word in s['answer'].lower().split():
        key = strip_accents(word)
        if key != word:                           # only words that DO have diacritics
            diacritics_map.setdefault(key, Counter())[word] += 1

# Keep only best accented match, BUT skip ambiguous keys where the plain form
# also appears frequently in training data (e.g. "ba" = 3 vs "bà" = lady).
_AMBIGUITY_THRESHOLD = 5          # tune: if plain form appears >= N times, skip
diacritics_map_final = {}
for k, ctr in diacritics_map.items():
    if _plain_word_freq.get(k, 0) >= _AMBIGUITY_THRESHOLD:
        continue                   # ambiguous — leave the word unchanged
    diacritics_map_final[k] = ctr.most_common(1)[0][0]

def restore_diacritics(text):
    words = text.lower().split()
    return ' '.join(diacritics_map_final.get(w, w) for w in words)

print(f'Data: {os.path.basename(train_json_path)} | Vocab: {VOCAB_SIZE} | Diacritics map: {len(diacritics_map_final)} words')

# ── Image transform ──────────────────────────────────────────
img_transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Model definitions ────────────────────────────────────────
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.backbone = nn.Sequential(*list(resnet50(weights=ResNet50_Weights.DEFAULT).children())[:-1])
        self.proj = nn.Linear(2048, embed_dim)
    def forward(self, x): return self.proj(self.backbone(x).flatten(1))

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.phobert = AutoModel.from_pretrained('vinai/phobert-base')
    def forward(self, input_ids, attention_mask):
        return self.phobert(input_ids=input_ids, attention_mask=attention_mask).pooler_output

class VQALSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=768, hidden_dim=512, num_layers=2):
        super().__init__()
        self.img_enc, self.txt_enc = ImageEncoder(embed_dim), TextEncoder()
        self.hidden_dim, self.num_layers = hidden_dim, num_layers
        self.context_proj_h = nn.Linear(embed_dim*2, hidden_dim*num_layers)
        self.context_proj_c = nn.Linear(embed_dim*2, hidden_dim*num_layers)
        self.ans_emb = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
    def _init_hidden(self, ctx):
        B = ctx.size(0)
        h = self.context_proj_h(ctx).view(B, self.num_layers, self.hidden_dim).permute(1,0,2).contiguous()
        c = self.context_proj_c(ctx).view(B, self.num_layers, self.hidden_dim).permute(1,0,2).contiguous()
        return h, c
    def forward(self, image, q_ids, q_mask, tgt_ids):
        ctx = torch.cat([self.img_enc(image), self.txt_enc(q_ids, q_mask)], dim=1)
        h0, c0 = self._init_hidden(ctx)
        return self.fc_out(self.lstm(self.ans_emb(tgt_ids), (h0, c0))[0])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=60):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class VQATransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=768, nhead=8, num_layers=4):
        super().__init__()
        self.img_enc, self.txt_enc = ImageEncoder(embed_dim), TextEncoder()
        self.ans_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=2048,
                                       dropout=0.1, batch_first=True), num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
    def forward(self, image, q_ids, q_mask, tgt_ids):
        memory = torch.cat([self.img_enc(image).unsqueeze(1), self.txt_enc(q_ids, q_mask).unsqueeze(1)], dim=1)
        tgt = self.pos_enc(self.ans_emb(tgt_ids))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        return self.fc_out(self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask, tgt_key_padding_mask=(tgt_ids==0)))

# ── Greedy decode for A1/A2 ───────────────────────────────────
@torch.no_grad()
def generate_answer_custom(model, img_t, q_ids, q_mask):
    model.eval()
    bos_id, eos_id = vocab['<BOS>'], vocab['<EOS>']
    generated = [bos_id]
    for _ in range(MAX_A_LEN - 1):
        tgt = torch.tensor([generated], device=device)
        next_id = model(img_t, q_ids, q_mask, tgt)[0, -1, :].argmax().item()
        if next_id == eos_id: break
        generated.append(next_id)
    return decode_answer_seq(generated)

# ── Load all models ───────────────────────────────────────────
models_loaded = {}
blip_name = 'Salesforce/blip-vqa-base'

for name, cls, path in [
    ('A1 — LSTM Decoder', VQALSTMDecoder, 'best_A1.pt'),
    ('A2 — Transformer Decoder', VQATransformerDecoder, 'best_A2.pt'),
]:
    p = os.path.join(BASE_DIR, path)
    if os.path.exists(p):
        print(f'Loading {name}...')
        m = cls(VOCAB_SIZE).to(device)
        m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        m.eval()
        models_loaded[name] = ('custom', m)

print('Loading BLIP...')
blip_proc = BlipProcessor.from_pretrained(blip_name)
blip_zs = BlipForQuestionAnswering.from_pretrained(blip_name).to(device).eval()
models_loaded['B1 — BLIP Zero-shot'] = ('blip_zs', blip_zs)

# Track which BLIP fine-tuned models are Vietnamese-native vs English-trained
blip_ft_is_vietnamese = {}   # key -> bool

for name, path, mtype in [
    ('B2 — BLIP Fine-tuned', 'best_B2.pt', 'blip_ft'),
    ('B2+RL — DPO Enhanced', 'best_B2_RL.pt', 'blip_ft'),
]:
    p = os.path.join(BASE_DIR, path)
    # Prefer Vietnamese-trained variants if available
    vi_path_map = {
        'best_B2.pt':    'best_B2_vi.pt',
        'best_B2_RL.pt': 'best_B2_RL_vi.pt',
    }
    vi_p = os.path.join(BASE_DIR, vi_path_map.get(path, ''))
    use_path, is_vi = (vi_p, True) if os.path.exists(vi_p) else (p, False)
    if os.path.exists(use_path):
        tag = '(VI)' if is_vi else '(EN)'
        print(f'Loading {name} {tag}...')
        m = BlipForQuestionAnswering.from_pretrained(blip_name)
        m.load_state_dict(torch.load(use_path, map_location=device, weights_only=True))
        models_loaded[name] = (mtype, m.to(device).eval())
        blip_ft_is_vietnamese[name] = is_vi

gc.collect()
if device.type == 'cuda': torch.cuda.empty_cache()
print(f'Loaded {len(models_loaded)} models')
for k, v in blip_ft_is_vietnamese.items():
    print(f'  {k}: {"Vietnamese native" if v else "English (translate pipeline)"}')

# Translator for B1
from deep_translator import GoogleTranslator
vi2en = GoogleTranslator(source='vi', target='en')
en2vi = GoogleTranslator(source='en', target='vi')
def translate_safe(tr, text):
    try:
        return tr.translate(text)
    except Exception:
        return text

def expand_blip_answer(raw_en: str, question_vi: str) -> str:
    """Convert short BLIP English output into a natural Vietnamese answer."""
    raw = raw_en.lower().strip()
    q   = question_vi.lower()
    # Expand yes/no to full sentences
    if raw in ('yes', 'yeah', 'true'):
        if 'không' in q:
            return 'Có, đúng vậy.'
        return 'Có.'
    if raw in ('no', 'nope', 'false'):
        if 'không' in q:
            return 'Không, món này không có.'
        return 'Không.'
    # Expand number answers — only wrap with context if clearly a counting question
    num_map = {'0':'không có','1':'một','2':'hai','3':'ba','4':'bốn',
               '5':'năm','6':'sáu','7':'bảy','8':'tám','9':'chín','10':'mười'}
    if raw in num_map:
        vi_num = num_map[raw]
        if any(kw in q for kw in ['bao nhiêu','mấy','số lượng']):
            # Detect unit from question
            unit_map = [
                ('miếng','miếng'),('cái','cái'),('bát','bát'),('đĩa','đĩa'),
                ('cuốn','cuốn'),('quả','quả'),('chiếc','chiếc'),('con','con'),
                ('bát','bát'),('lát','lát'),('phần','phần'),
            ]
            unit = 'phần'
            for kw, u in unit_map:
                if kw in q:
                    unit = u
                    break
            return f'Có {vi_num} {unit}.'
        return vi_num
    # For other answers: translate to Vietnamese
    try:
        vi = en2vi.translate(raw_en)
        return vi if vi and len(vi) > 1 else raw_en
    except Exception:
        return raw_en


# ── Post-processing helpers ────────────────────────────────────
def clean_blip_output(text: str, max_words: int = 12) -> str:
    """Remove repetitive n-gram loops and trim to reasonable length."""
    words = text.strip().split()
    if not words:
        return text
    # Detect and cut at first 3-gram repetition
    seen_trigrams = set()
    for i in range(len(words) - 2):
        tg = (words[i], words[i+1], words[i+2])
        if tg in seen_trigrams:
            words = words[:i]          # truncate at loop start
            break
        seen_trigrams.add(tg)
    # Hard cap on word count
    words = words[:max_words]
    return ' '.join(words)

def sanitize_b1_output(text: str, question_vi: str) -> str:
    """B1 (zero-shot, translated) sometimes returns bare digits — add context."""
    t = text.strip()
    if t.isdigit():                    # e.g. "6" with no unit
        # Heuristic: questions about quantity → wrap with unit phrase
        if any(kw in question_vi.lower() for kw in ["bao nhiêu", "mấy", "số lượng"]):
            return f"khoảng {t} phần"
        return t
    return t

# ── Inference ─────────────────────────────────────────────────
@torch.no_grad()
def run_model(image_pil, question, model_key):
    mtype, model = models_loaded[model_key]
    if mtype == 'custom':
        img_t = img_transform(image_pil).unsqueeze(0).to(device)
        enc = encode_text(question)
        answer = generate_answer_custom(model, img_t, enc['input_ids'].to(device), enc['attention_mask'].to(device))
    elif mtype == 'blip_zs':
        q_en = translate_safe(vi2en, question)
        inputs = blip_proc(images=image_pil, text=q_en, return_tensors='pt').to(device)
        out = model.generate(**inputs, max_length=30, num_beams=5, min_length=2)
        raw = blip_proc.decode(out[0], skip_special_tokens=True)
        translated = expand_blip_answer(raw, question)
        answer = sanitize_b1_output(translated, question)
    else:
        if blip_ft_is_vietnamese.get(model_key, False):
            # ── Vietnamese fine-tuned model: feed VI question directly ──
            inputs = blip_proc(images=image_pil, text=question,
                               return_tensors='pt').to(device)
            out = model.generate(
                **inputs, max_length=48, num_beams=5,
                min_length=3, repetition_penalty=1.3,
                length_penalty=1.1, early_stopping=True,
            )
            raw_vi = blip_proc.decode(out[0], skip_special_tokens=True).strip()
            answer = clean_blip_output(raw_vi, max_words=20)
        else:
            # ── Legacy English fine-tuned model: VI→EN→BLIP→VI ──
            q_en = translate_safe(vi2en, question)
            inputs = blip_proc(images=image_pil, text=q_en,
                               return_tensors='pt').to(device)
            out = model.generate(**inputs, max_length=40, num_beams=5,
                                 min_length=3, repetition_penalty=1.2,
                                 length_penalty=1.0)
            raw_en = blip_proc.decode(out[0], skip_special_tokens=True).strip()
            raw_en = clean_blip_output(raw_en, max_words=15)
            raw_vi = translate_safe(en2vi, raw_en)
            answer = clean_blip_output(raw_vi, max_words=15)
    return restore_diacritics(answer)

def predict(image, question, model_key):
    if image is None: return "Vui lòng tải lên ảnh món ăn."
    if not question or not question.strip(): return "Vui lòng nhập câu hỏi."
    if not isinstance(image, Image.Image): image = Image.fromarray(image)
    return run_model(image.convert('RGB'), question, model_key)

def predict_all(image, question):
    if image is None: return "Vui lòng tải lên ảnh món ăn."
    if not question or not question.strip(): return "Vui lòng nhập câu hỏi."
    if not isinstance(image, Image.Image): image = Image.fromarray(image)
    image = image.convert('RGB')
    lines = []
    for name in models_loaded:
        try:
            ans = run_model(image, question, name)
        except Exception as e:
            ans = f"(Lỗi: {e})"
        icon = {"A1":"🔵","A2":"🟣","B1":"🟡","B2":"🟢","B2+RL":"🔴"}.get(name[:2], "⚪")
        lines.append(f"{icon} **{name}**\n\n> {ans}\n")
    return "\n---\n".join(lines)

# ── GRADIO UI ─────────────────────────────────────────────────
import gradio as gr

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', system-ui, sans-serif !important; }
.gradio-container { max-width: 960px !important; margin: auto !important; }
footer { display: none !important; }
.header-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
    border-radius: 20px; padding: 40px 32px 32px; margin-bottom: 24px;
    text-align: center; position: relative; overflow: hidden;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
}
.header-card::before {
    content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(99,102,241,0.15) 0%, transparent 50%),
                radial-gradient(circle at 70% 50%, rgba(244,63,94,0.1) 0%, transparent 50%);
}
.header-card h1 {
    font-size: 2.2em; font-weight: 800; margin: 0;
    background: linear-gradient(135deg, #818cf8, #c084fc, #fb7185);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    position: relative;
}
.header-card p { color: #94a3b8; font-size: 1em; margin: 8px 0 0; position: relative; }
.header-card .badges { margin-top: 16px; position: relative; }
.header-card .badges span {
    display: inline-block; padding: 4px 14px; margin: 4px;
    border-radius: 20px; font-size: 0.78em; font-weight: 600;
    background: rgba(255,255,255,0.08); color: #cbd5e1;
    border: 1px solid rgba(255,255,255,0.1);
}
.input-card, .output-card {
    background: #ffffff; border-radius: 16px;
    border: 1px solid #e2e8f0; box-shadow: 0 4px 20px rgba(0,0,0,0.04);
}
.section-title {
    font-size: 0.82em; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.5px; color: #6366f1; margin: 0 0 12px 0;
}
.answer-display textarea {
    font-size: 1.3em !important; font-weight: 600 !important;
    color: #1e293b !important; background: linear-gradient(135deg, #f0fdf4, #ecfdf5) !important;
    border: 2px solid #86efac !important; border-radius: 14px !important;
    padding: 20px !important; line-height: 1.6 !important;
}
.compare-box { max-height: 500px; overflow-y: auto; }
.suggest-btn { border-radius: 24px !important; font-size: 0.82em !important;
    font-weight: 500 !important; border: 1px solid #e2e8f0 !important;
    transition: all 0.2s !important; }
.suggest-btn:hover { border-color: #6366f1 !important; color: #6366f1 !important;
    background: #eef2ff !important; transform: translateY(-1px) !important; }
.primary-btn { border-radius: 14px !important; font-size: 1.05em !important;
    font-weight: 700 !important; padding: 12px 0 !important;
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.4) !important;
    transition: all 0.3s !important; }
.primary-btn:hover { transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.5) !important; }
.secondary-btn { border-radius: 14px !important; font-weight: 600 !important; }
"""

model_choices = list(models_loaded.keys())

with gr.Blocks(title="VQA Tiếng Việt") as demo:
    gr.HTML("""<div class="header-card">
        <h1>Visual Question Answering</h1>
        <p>Hệ thống Hỏi Đáp trên Ảnh — Món Ăn Việt Nam</p>
        <p style="font-size:0.85em; color:#64748b;">Đồ án cuối kỳ môn Học Sâu</p>
        <div class="badges">
            <span>🔵 A1 LSTM</span><span>🟣 A2 Transformer</span>
            <span>🟡 B1 Zero-shot</span><span>🟢 B2 Fine-tuned</span>
            <span>🔴 B2+RL DPO</span>
        </div>
    </div>""")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.HTML('<p class="section-title">📷 Tải ảnh lên</p>')
            img_input = gr.Image(label=None, type="pil", height=280,
                                 sources=["upload", "clipboard"])
            gr.HTML('<p class="section-title" style="margin-top:16px">❓ Câu hỏi</p>')
            question_input = gr.Textbox(label=None, placeholder="Ví dụ: Đây là món gì?", lines=2)
            gr.HTML('<p class="section-title" style="margin-top:12px">🧠 Mô hình</p>')
            model_selector = gr.Dropdown(choices=model_choices, value=model_choices[-1] if model_choices else None,
                                         label=None, interactive=True)
            with gr.Row():
                btn_single = gr.Button("Trả lời", variant="primary", scale=2, elem_classes=["primary-btn"])
                btn_all = gr.Button("So sánh 5 model", variant="secondary", scale=1, elem_classes=["secondary-btn"])

        with gr.Column(scale=1):
            gr.HTML('<p class="section-title">💬 Câu trả lời</p>')
            answer_output = gr.Textbox(label=None, lines=3, interactive=False, elem_classes=["answer-display"])
            gr.HTML('<p class="section-title" style="margin-top:16px">📊 So sánh tất cả mô hình</p>')
            compare_output = gr.Markdown(value="*Nhấn \"So sánh 5 model\" để xem kết quả từ tất cả mô hình.*",
                                         elem_classes=["compare-box"])

    gr.HTML('<p class="section-title" style="margin-top:20px">💡 Câu hỏi gợi ý</p>')
    with gr.Row():
        for q in ["Đây là món gì?", "Món này có nước không?", "Có bao nhiêu thành phần?", "Món này có cay không?"]:
            gr.Button(q, size="sm", elem_classes=["suggest-btn"]).click(fn=lambda x=q: x, outputs=question_input)
    with gr.Row():
        for q in ["Màu sắc của món ăn là gì?", "Trong bát có rau không?", "Món này ăn nóng hay lạnh?", "Món này làm từ gì?"]:
            gr.Button(q, size="sm", elem_classes=["suggest-btn"]).click(fn=lambda x=q: x, outputs=question_input)

    gr.HTML("""<div style="text-align:center; margin-top:24px; padding:16px;
        background:#f8fafc; border-radius:12px; border:1px solid #e2e8f0;">
        <p style="color:#64748b; font-size:0.85em; margin:0;">
            <b>Kiến trúc:</b> A1/A2 = ResNet50 + PhoBERT + Decoder &nbsp;|&nbsp;
            B1/B2 = BLIP VQA &nbsp;|&nbsp; B2+RL = DPO Reinforcement Learning<br>
            <b>Dataset:</b> Vietnamese Foods (Kaggle) — 15000 cặp QA &nbsp;|&nbsp;
            <b>Framework:</b> PyTorch + HuggingFace
        </p>
    </div>""")

    btn_single.click(fn=predict, inputs=[img_input, question_input, model_selector], outputs=answer_output)
    btn_all.click(fn=predict_all, inputs=[img_input, question_input], outputs=compare_output)
    question_input.submit(fn=predict, inputs=[img_input, question_input, model_selector], outputs=answer_output)

if __name__ == '__main__':
    print('\nStarting VQA Demo -> http://localhost:7860')
    demo.launch(server_name='0.0.0.0', server_port=7860, share=False, show_error=True)
