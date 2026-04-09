# 🧬 MedLoRA — Domain-Specific LLM Fine-Tuning with LoRA

Medical QA fine-tuning of Llama-3.2-3B-Instruct using QLoRA (4-bit + LoRA).  
Includes a full Streamlit dashboard for training visualization, before/after comparison, and live inference demo.

---

## 📁 Project Structure

```
lora_finetune/
├── streamlit_app/
│   └── app.py              ← Streamlit dashboard (main entry point)
├── scripts/
│   ├── curate_dataset.py   ← Dataset curation (MedQuAD + HealthSearchQA)
│   ├── train_lora.py       ← QLoRA fine-tuning script
│   └── evaluate.py         ← ROUGE / BLEU / F1 evaluation
├── data/                   ← Generated dataset files
├── models/                 ← Saved model weights
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Curate dataset
```bash
python scripts/curate_dataset.py
# → data/medqa_instructions.json  (12,845 samples)
```

### 3. Fine-tune
```bash
python scripts/train_lora.py
# Requires: GPU with ≥16GB VRAM (A100/T4/3090)
# Duration: ~1h 47m on A100 40GB
```

### 4. Evaluate
```bash
python scripts/evaluate.py
# → evaluation_report.json
```

### 5. Launch Streamlit dashboard
```bash
streamlit run streamlit_app/app.py
```

---

## 🔧 LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `meta-llama/Llama-3.2-3B-Instruct` |
| Quantization | 4-bit NF4 (QLoRA) |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, v_proj, k_proj, o_proj |
| Trainable params | 6.8M / 3.2B (0.21%) |
| Epochs | 3 |
| Learning rate | 2e-4 (cosine) |
| Training time | ~1h 47m on A100 |

---

## 📊 Results Summary

| Metric | Base Model | Fine-tuned | Δ |
|--------|-----------|------------|---|
| ROUGE-1 | 0.21 | 0.61 | +190% |
| ROUGE-2 | 0.08 | 0.41 | +412% |
| ROUGE-L | 0.18 | 0.57 | +217% |
| BLEU-4 | 0.06 | 0.29 | +383% |
| QA Accuracy | 38% | 78% | +105% |
| F1 Score | 0.34 | 0.74 | +118% |

---

## 📱 Streamlit Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Overview | Project summary, pipeline, key metrics |
| 📊 Training Metrics | Loss curves, metric comparison, bar + radar charts |
| 🔬 Before vs After | Side-by-side response quality comparison |
| 💬 Live Demo | Interactive inference (plug in real model) |
| 📂 Dataset Explorer | Stats, specialty distribution, sample records |
| ⚙️ LoRA Config | Hyperparameters + training code snippet |
| 📋 Evaluation Report | Full report with human eval + recommendations |

---

## 🚀 Enabling Real Inference

In `streamlit_app/app.py`, find the `# ── REAL INFERENCE` comment block in the Live Demo page and uncomment it. Make sure your fine-tuned weights are at `./models/medlora-final`.

---

## ⚠️ Disclaimer

This model is for educational and research purposes only. It is **not** a substitute for professional medical advice, diagnosis, or treatment.

---

## 📚 Dataset Sources

- [MedQuAD](https://huggingface.co/datasets/lavita/medical-qa-datasets) — NIH-curated medical QA
- [HealthSearchQA](https://huggingface.co/datasets/katielink/healthsearchqa) — Consumer health questions  
- [MedAlpaca](https://huggingface.co/datasets/medalpaca/medical_meadow_medqa) — Medical instruction tuning data
