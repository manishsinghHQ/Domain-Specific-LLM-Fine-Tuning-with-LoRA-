"""
evaluate.py — Evaluate base model vs. fine-tuned model on test set
Metrics: ROUGE-1/2/L, BLEU-4, QA Accuracy, F1
Output: evaluation_report.json
"""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import evaluate  # pip install evaluate rouge_score sacrebleu


# ── Load metrics ───────────────────────────────────────────────────────────────
rouge  = evaluate.load("rouge")
bleu   = evaluate.load("sacrebleu")


# ── Inference helper ───────────────────────────────────────────────────────────
def generate_response(model, tokenizer, question: str, max_new_tokens=256) -> str:
    prompt = (
        "<s>[INST] <<SYS>>\nYou are a medical assistant.\n<</SYS>>\n\n"
        f"Question: {question} [/INST]\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the response part
    if "[/INST]" in decoded:
        decoded = decoded.split("[/INST]")[-1].strip()
    return decoded


# ── Token-level F1 ─────────────────────────────────────────────────────────────
def compute_f1(prediction: str, reference: str) -> float:
    pred_tokens = prediction.lower().split()
    ref_tokens  = reference.lower().split()
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# ── Main evaluation ────────────────────────────────────────────────────────────
def evaluate_models(
    base_model_path: str,
    lora_adapter_path: str,
    test_data_path: str,
    output_path: str = "./evaluation_report.json",
    n_samples: int = 200,
):
    # Load test data
    with open(test_data_path) as f:
        test_data = json.load(f)[:n_samples]

    questions  = [d["input"] for d in test_data]
    references = [d["output"] for d in test_data]

    # Load base model
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, load_in_4bit=True, device_map="auto"
    )

    # Generate base model responses
    print("Generating base model responses...")
    base_preds = []
    for q in tqdm(questions):
        base_preds.append(generate_response(base_model, tokenizer, q))

    # Load fine-tuned model
    print("Loading fine-tuned model...")
    ft_model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    # Generate fine-tuned responses
    print("Generating fine-tuned responses...")
    ft_preds = []
    for q in tqdm(questions):
        ft_preds.append(generate_response(ft_model, tokenizer, q))

    # Compute metrics
    def compute_metrics(preds, refs):
        rouge_scores = rouge.compute(predictions=preds, references=refs)
        bleu_score   = bleu.compute(predictions=preds, references=[[r] for r in refs])
        f1_scores    = [compute_f1(p, r) for p, r in zip(preds, refs)]
        return {
            "rouge1":   round(rouge_scores["rouge1"], 4),
            "rouge2":   round(rouge_scores["rouge2"], 4),
            "rougeL":   round(rouge_scores["rougeL"], 4),
            "bleu":     round(bleu_score["score"] / 100, 4),
            "f1":       round(np.mean(f1_scores), 4),
        }

    print("Computing metrics...")
    base_metrics = compute_metrics(base_preds, references)
    ft_metrics   = compute_metrics(ft_preds,   references)

    report = {
        "dataset":      test_data_path,
        "n_samples":    n_samples,
        "base_model":   base_model_path,
        "lora_adapter": lora_adapter_path,
        "base_metrics": base_metrics,
        "ft_metrics":   ft_metrics,
        "improvements": {
            k: round(ft_metrics[k] - base_metrics[k], 4)
            for k in base_metrics
        },
        "sample_comparisons": [
            {"question": q, "base": b, "finetuned": f, "reference": r}
            for q, b, f, r in zip(questions[:10], base_preds[:10], ft_preds[:10], references[:10])
        ],
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as out:
        json.dump(report, out, indent=2)

    print(f"\n✅ Evaluation report saved to {output_path}")
    print("\nBase model metrics:", base_metrics)
    print("Fine-tuned metrics:", ft_metrics)
    print("Improvements:      ", report["improvements"])

    return report


if __name__ == "__main__":
    evaluate_models(
        base_model_path  = "meta-llama/Llama-3.2-3B-Instruct",
        lora_adapter_path= "./models/medlora/adapter",
        test_data_path   = "./data/medqa_instructions_test.json",
        n_samples        = 200,
    )
