# Deep Learning Final Project

This repository contains the source code and demo applications for the Deep Learning final project. The project includes two vision-language tasks:

- Task 1: Vietnamese Visual Question Answering for Vietnamese food images.
- Task 2: Chest X-ray report generation using both a traditional encoder-decoder baseline and a pretrained BLIP model.

Large datasets and model checkpoints are not committed to GitHub because of file size limits. They should be submitted separately through Google Drive or HuggingFace Hub.

## Members

- Le Nhat Huy - 523H0138
- Ta Duong - 523H0131

## Repository Structure

```text
.
├── README.md
├── TASK1
│   ├── Task1.ipynb
│   ├── demo_app.py
│   ├── requirements.txt
│   ├── vqa_dataset.json
│   └── vqa_dataset_flattened.json
└── TASK2
    ├── Food_Classifier.ipynb
    ├── xray_webapp
    │   ├── app.py
    │   ├── model_def.py
    │   ├── requirements.txt
    │   ├── templates/
    │   └── static/
    └── blip_webapp
        ├── app.py
        ├── evaluate_test.py
        ├── requirements.txt
        ├── templates/
        └── static/
```

## Task 1 - Vietnamese Visual Question Answering

Task 1 builds a Vietnamese VQA system in the Vietnamese food domain. The system receives an image and a Vietnamese question, then returns a short Vietnamese answer.

### Dataset

The dataset contains Vietnamese food images and image-question-answer pairs.

- Domain: Vietnamese dishes.
- Number of classes: 30 food classes.
- Number of images: about 3000 images.
- Number of QA pairs in the flattened JSON files: 15000.
- Question types include recognition, yes/no, counting, attribute, and spatial questions.

The image dataset folder is not stored in this repository. To run the full demo with sample images, place the extracted dataset folder here:

```text
TASK1/vqa_dataset_final/
```

### Model Configurations

The project compares the following configurations:

- A1: ResNet50 + PhoBERT + LSTM decoder.
- A2: ResNet50 + PhoBERT + Transformer decoder.
- B1: BLIP zero-shot with Vietnamese-English translation.
- B2: Fine-tuned BLIP.
- B2+RL: BLIP improved with preference optimization / DPO.

Task 1 checkpoints are not committed to GitHub. Place them in `TASK1/` before running the full demo:

```text
TASK1/best_A1.pt
TASK1/best_A2.pt
TASK1/best_B2.pt
TASK1/best_B2_RL.pt
```

If Vietnamese-native BLIP checkpoints are available, the demo also supports:

```text
TASK1/best_B2_vi.pt
TASK1/best_B2_RL_vi.pt
```

### Run Task 1 Demo

```bash
cd TASK1
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python demo_app.py
```

Open:

```text
http://localhost:7860
```

Notes:

- The demo always loads BLIP zero-shot.
- Custom A1/A2 and fine-tuned B2 models are loaded only if their checkpoint files exist.
- `demo_app.py` uses `vqa_train.json` if available. Otherwise, it falls back to `vqa_dataset_flattened.json`.

## Task 2 - Chest X-ray Report Generation

Task 2 focuses on generating radiology-style reports from chest X-ray images. It includes two demo applications for comparison.

### Approach 1 - DenseNet121 + LSTM

Folder:

```text
TASK2/xray_webapp
```

This is the baseline model. It uses DenseNet121 as the visual encoder and an LSTM decoder with attention to generate text.

Required checkpoint:

```text
TASK2/xray_webapp/model/final_model_lstm.pth
```

Run:

```bash
cd TASK2/xray_webapp
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open:

```text
http://localhost:5001
```

### Approach 2 - Fine-tuned BLIP

Folder:

```text
TASK2/blip_webapp
```

This is the main pretrained multimodal approach. It fine-tunes BLIP for chest X-ray report generation and compares against the DenseNet + LSTM baseline.

Required checkpoint:

```text
TASK2/blip_webapp/model/best_blip.pt
```

Optional checkpoint:

```text
TASK2/blip_webapp/model/blip_final.pt
```

Run:

```bash
cd TASK2/blip_webapp
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open:

```text
http://localhost:5002
```

## Evaluation

The report discusses both strict and semantic evaluation metrics:

- Exact Match / VQA Accuracy for short VQA answers.
- BLEU, ROUGE-L, and METEOR for text overlap.
- BERTScore for semantic similarity.
- LLM-as-a-judge for qualitative evaluation.

Task 2 also compares DenseNet + LSTM against fine-tuned BLIP using text generation metrics.

## Files Not Included in GitHub

The following files are intentionally excluded by `.gitignore`:

- Python virtual environments: `venv/`, `.venv/`.
- Local upload folders: `static/uploads/`.
- Large datasets: `TASK1/vqa_dataset_final/`, `TASK1/vqa_dataset_final.zip`.
- Model checkpoints: `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, `*.bin`.
- Duplicate local working copies such as `TASK2/* copy/`.
- macOS temporary files such as `.DS_Store`.
