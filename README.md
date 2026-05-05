# Deep Learning Final Project

Final project for the Deep Learning course.

## Members

- Le Nhat Huy - 523H0138
- Ta Duong - 523H0131

## Tasks

### Task 1 - Vietnamese Visual Question Answering

Domain: Vietnamese food images.

Main files:

- `TASK1/Task1.ipynb`
- `TASK1/demo_app.py`
- `TASK1/vqa_dataset.json`
- `TASK1/vqa_dataset_flattened.json`

Large dataset images are not stored in this GitHub repository. Submit them separately through Google Drive or HuggingFace Hub.

Model checkpoints are not stored in this GitHub repository. Put Task 1 checkpoints such as `best_A1.pt`, `best_A2.pt`, `best_B2.pt`, and `best_B2_RL.pt` in `TASK1/` before running the full demo.

### Task 2 - Chest X-ray Report Generation

Two Flask demos are included:

- `TASK2/xray_webapp`: DenseNet121 + LSTM baseline
- `TASK2/blip_webapp`: Fine-tuned BLIP model

Model checkpoints are not stored in this GitHub repository. Download/copy them into each app's `model/` folder before running.

## Run Task 1 Demo

```bash
cd TASK1
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python demo_app.py
```

Open: <http://localhost:7860>

## Run Task 2 Demos

DenseNet + LSTM:

```bash
cd TASK2/xray_webapp
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open: <http://localhost:5001>

BLIP:

```bash
cd TASK2/blip_webapp
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open: <http://localhost:5002>

## Submission Notes

Submit these separately with the GitHub link:

- Final report
- Slides
- Demo video
- Dataset archive
- Model checkpoints
