import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice

HF_MODEL_REPO = "Shitanshu06/mcq-deberta-v3-best-v2"
OPTION_COLUMNS = ["A", "B", "C", "D", "E"]
MAX_LENGTH = 192


class MCQSolver:
    def __init__(self, model_dir=HF_MODEL_REPO, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForMultipleChoice.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, prompt: str, options: list):
        assert len(options) == len(OPTION_COLUMNS), f"Expected {len(OPTION_COLUMNS)} options"

        encoded = self.tokenizer(
            [prompt] * len(options),
            options,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.unsqueeze(0).to(self.device) for k, v in encoded.items()}
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        ranked_idx = np.argsort(probs)[::-1]
        ranked_labels = [OPTION_COLUMNS[i] for i in ranked_idx]

        return {
            "top3": ranked_labels[:3],
            "prediction": ranked_labels[0],
            "probabilities": {OPTION_COLUMNS[i]: float(probs[i]) for i in range(len(options))},
        }
