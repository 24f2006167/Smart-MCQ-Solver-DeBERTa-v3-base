import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMultipleChoice

# ── Config ──────────────────────────────────────────────────────────────────
HF_MODEL_REPO = "Shitanshu06/mcq-deberta-v3-best-v2"
OPTION_LABELS = ["A", "B", "C", "D", "E"]
MAX_LENGTH = 256

# ── Model loader (cached globally) ──────────────────────────────────────────
_model = None
_tokenizer = None
_device = None


def load_model(force_reload=False):
    global _model, _tokenizer, _device
    if _model is None or force_reload:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading tokenizer & model from '{HF_MODEL_REPO}' on {_device}...")
        _tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_REPO)
        _model = AutoModelForMultipleChoice.from_pretrained(HF_MODEL_REPO)
        _model.to(_device)
        _model.eval()
        print("Model loaded successfully!")
    return _model, _tokenizer, _device


# ── Inference ────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(prompt, opt_a, opt_b, opt_c, opt_d, opt_e):
    options = [opt_a, opt_b, opt_c, opt_d, opt_e]

    if not prompt or not prompt.strip():
        return (
            "⚠️ **Please enter a question.**",
            "N/A",
            {lb: 0.20 for lb in OPTION_LABELS},
        )
    if not all(o and o.strip() for o in options):
        return (
            "⚠️ **Please fill in all 5 options (A, B, C, D, E).**",
            "N/A",
            {lb: 0.20 for lb in OPTION_LABELS},
        )

    model, tokenizer, device = load_model()

    # Pair prompt with each option
    encoded = tokenizer(
        [prompt] * 5,
        options,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    inputs = {k: v.unsqueeze(0).to(device) for k, v in encoded.items()}
    
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    ranked_idx = np.argsort(probs)[::-1]
    ranked_labels = [OPTION_LABELS[i] for i in ranked_idx]

    top3_str = " → ".join(ranked_labels[:3])
    best_idx = ranked_idx[0]
    best_label = OPTION_LABELS[best_idx]
    best_text = options[best_idx]
    best_prob = probs[best_idx] * 100

    prediction_markdown = f"""### 🏆 Predicted Answer: **Option {best_label}** — *{best_text}*
    
**Confidence:** `{best_prob:.1f}%`  
**Top-3 Order (MAP@3):** `{top3_str}`
"""

    prob_dict = {
        f"Option {OPTION_LABELS[i]} ({options[i][:25]}...)": float(probs[i])
        for i in range(5)
    }

    return prediction_markdown, top3_str, prob_dict


# ── High-Contrast Readable CSS ────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; }

/* Main Container */
.gradio-container {
    background: #0f172a !important;
    color: #f8fafc !important;
}

/* Hero Header Banner */
#hero-banner {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #0284c7 100%);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 16px;
    text-align: center;
    box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.4);
}
#hero-banner h1 {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    margin: 0 0 6px 0 !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}
#hero-banner p {
    color: #f1f5f9 !important;
    font-size: 1rem !important;
    margin: 0 !important;
    font-weight: 500;
}

/* Card Containers */
.section-card {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 14px !important;
    padding: 20px !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
}

/* All Headings & Labels - Readable High Contrast */
h1, h2, h3, h4, h5, h6 {
    color: #f8fafc !important;
    font-weight: 700 !important;
}
.section-heading, .section-heading h3 {
    color: #38bdf8 !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* Labels on Textboxes */
label, label span, .gr-form label span {
    color: #93c5fd !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}

/* Input Fields - Deep contrasting background with crisp white text */
textarea, input[type="text"] {
    background-color: #0f172a !important;
    border: 1.5px solid #475569 !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.3) !important;
    background-color: #1e1b4b !important;
}

/* Predict Button */
#predict-btn {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    padding: 14px !important;
    box-shadow: 0 4px 18px rgba(99, 102, 241, 0.4) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
#predict-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.6) !important;
}

/* Result Markdown Box */
#result-box {
    background: rgba(16, 185, 129, 0.15) !important;
    border: 1.5px solid #10b981 !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    color: #f0fdf4 !important;
}
#result-box h3 {
    color: #4ade80 !important;
    font-size: 1.2rem !important;
    margin-top: 0 !important;
}

/* Top 3 Box */
#top3-box textarea {
    background-color: #0c4a6e !important;
    border-color: #0284c7 !important;
    color: #7dd3fc !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    text-align: center !important;
}

/* Confidence Score Widget */
.gr-label {
    background: #0f172a !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    color: #ffffff !important;
}
.gr-label .label-item {
    color: #ffffff !important;
}

/* Helper Notes */
#hint-text, #hint-text p, blockquote, blockquote p {
    color: #cbd5e1 !important;
    font-size: 0.85rem !important;
}

/* Native Examples Table */
.gr-examples {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 14px !important;
    padding: 16px !important;
    margin-top: 16px !important;
}
.gr-examples table {
    color: #e2e8f0 !important;
}
.gr-examples tr:hover {
    background: #334155 !important;
    cursor: pointer !important;
}
"""

# ── Header HTML ───────────────────────────────────────────────────────────────
HERO_HTML = """
<div id="hero-banner">
  <h1>🧠 Smart MCQ Solver</h1>
  <p>DeBERTa-v3 Multiple Choice Question Answering &nbsp;·&nbsp; IIT Madras BS DS Project</p>
</div>
"""

# ── Build Gradio App ──────────────────────────────────────────────────────────
with gr.Blocks(
    title="Smart MCQ Solver — DeBERTa-v3",
    css=CUSTOM_CSS,
) as demo:

    gr.HTML(HERO_HTML)

    with gr.Row():
        # Left Column — Question & Options Input
        with gr.Column(scale=3, elem_classes=["section-card"]):
            gr.Markdown("### 📝 QUESTION & OPTIONS", elem_classes=["section-heading"])

            prompt_input = gr.Textbox(
                label="Question / Prompt",
                placeholder="Enter your multiple-choice question here...",
                lines=3,
                elem_id="prompt",
            )

            with gr.Row():
                opt_a = gr.Textbox(label="Option A", placeholder="First choice", elem_id="opt_a")
                opt_b = gr.Textbox(label="Option B", placeholder="Second choice", elem_id="opt_b")

            with gr.Row():
                opt_c = gr.Textbox(label="Option C", placeholder="Third choice", elem_id="opt_c")
                opt_d = gr.Textbox(label="Option D", placeholder="Fourth choice", elem_id="opt_d")

            opt_e = gr.Textbox(label="Option E", placeholder="Fifth choice", elem_id="opt_e")

            predict_btn = gr.Button("🔍 Predict Answer", variant="primary", size="lg", elem_id="predict-btn")

        # Right Column — Model Predictions & Confidence Scores
        with gr.Column(scale=2, elem_classes=["section-card"]):
            gr.Markdown("### 📊 MODEL OUTPUT", elem_classes=["section-heading"])

            prediction_out = gr.Markdown(
                value="*Select an example below or enter a question and click Predict Answer.*",
                elem_id="result-box",
            )

            top3_out = gr.Textbox(
                label="Top-3 Ranking (MAP@3 Order)",
                interactive=False,
                elem_id="top3-box",
            )

            prob_out = gr.Label(
                label="Confidence Scores — All 5 Options",
                num_top_classes=5,
            )

            gr.Markdown(
                "> **Note:** Probabilities are calculated using softmax logits from `DebertaV2ForMultipleChoice`.",
                elem_id="hint-text",
            )

    # Examples list
    gr.Examples(
        examples=[
            [
                "Which organelle is responsible for producing energy in the form of ATP through cellular respiration?",
                "Nucleus",
                "Ribosome",
                "Mitochondria",
                "Golgi apparatus",
                "Endoplasmic reticulum",
            ],
            [
                "Which of the following is NOT a programming language?",
                "Python",
                "Java",
                "HTML",
                "C++",
                "Ruby",
            ],
            [
                "Which element has the highest electronegativity on the Pauling scale?",
                "Oxygen",
                "Chlorine",
                "Nitrogen",
                "Fluorine",
                "Bromine",
            ],
            [
                "Which data structure follows the Last-In-First-Out (LIFO) principle?",
                "Queue",
                "Linked List",
                "Stack",
                "Binary Tree",
                "Hash Table",
            ],
            [
                "Which city serves as the capital of Australia?",
                "Sydney",
                "Melbourne",
                "Brisbane",
                "Canberra",
                "Perth",
            ],
        ],
        inputs=[prompt_input, opt_a, opt_b, opt_c, opt_d, opt_e],
        outputs=[prediction_out, top3_out, prob_out],
        fn=predict,
        cache_examples=False,
        label="💡 Click any example question below to load & test immediately:",
    )

    # Footer
    gr.HTML("""
    <div style="
        margin-top:20px;
        padding:16px;
        background:#1e293b;
        border:1px solid #334155;
        border-radius:12px;
        color:#94a3b8;
        font-size:0.85rem;
        text-align:center;
    ">
        <strong style="color:#38bdf8">Model:</strong> DeBERTa-v3-base &nbsp;|&nbsp;
        <strong style="color:#38bdf8">Fine-tuned Repository:</strong> Shitanshu06/mcq-deberta-v3-best-v2 &nbsp;|&nbsp;
        <strong style="color:#38bdf8">Author:</strong> Shitanshu Chaurasiya (24F2006167)
    </div>
    """)

    # Event binding
    predict_btn.click(
        fn=predict,
        inputs=[prompt_input, opt_a, opt_b, opt_c, opt_d, opt_e],
        outputs=[prediction_out, top3_out, prob_out],
    )

if __name__ == "__main__":
    load_model(force_reload=True)
    demo.launch()