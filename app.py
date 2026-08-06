import gradio as gr
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice

# ── Config ──────────────────────────────────────────────────────────────────
HF_MODEL_REPO = "Shitanshu06/mcq-deberta-v3-best-v2"
OPTION_LABELS = ["A", "B", "C", "D", "E"]
MAX_LENGTH = 192

# ── Model loader (cached globally) ──────────────────────────────────────────
_model = None
_tokenizer = None
_device = None


def load_model():
    global _model, _tokenizer, _device
    if _model is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_REPO)
        _model = AutoModelForMultipleChoice.from_pretrained(HF_MODEL_REPO)
        _model.to(_device)
        _model.eval()
    return _model, _tokenizer, _device


# ── Example Questions ────────────────────────────────────────────────────────
EXAMPLES = [
    {
        "label": "🔬 Biology — Cell Organelle",
        "question": "Which organelle is responsible for producing energy in the form of ATP through cellular respiration?",
        "A": "Nucleus",
        "B": "Ribosome",
        "C": "Mitochondria",
        "D": "Golgi apparatus",
        "E": "Endoplasmic reticulum",
    },
    {
        "label": "🧪 Chemistry — Periodic Table",
        "question": "Which element has the highest electronegativity on the Pauling scale?",
        "A": "Oxygen",
        "B": "Chlorine",
        "C": "Nitrogen",
        "D": "Fluorine",
        "E": "Bromine",
    },
    {
        "label": "📐 Mathematics — Calculus",
        "question": "What is the derivative of sin(x) with respect to x?",
        "A": "-sin(x)",
        "B": "cos(x)",
        "C": "-cos(x)",
        "D": "tan(x)",
        "E": "sec²(x)",
    },
    {
        "label": "💻 Computer Science — Data Structures",
        "question": "Which data structure follows the Last-In-First-Out (LIFO) principle?",
        "A": "Queue",
        "B": "Linked List",
        "C": "Stack",
        "D": "Binary Tree",
        "E": "Hash Table",
    },
    {
        "label": "🌍 Geography — World Capitals",
        "question": "Which city serves as the capital of Australia?",
        "A": "Sydney",
        "B": "Melbourne",
        "C": "Brisbane",
        "D": "Canberra",
        "E": "Perth",
    },
    {
        "label": "⚛️ Physics — Thermodynamics",
        "question": "According to the second law of thermodynamics, which quantity always increases in an isolated system?",
        "A": "Temperature",
        "B": "Pressure",
        "C": "Entropy",
        "D": "Enthalpy",
        "E": "Internal energy",
    },
]


# ── Inference ────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(prompt, opt_a, opt_b, opt_c, opt_d, opt_e):
    options = [opt_a, opt_b, opt_c, opt_d, opt_e]

    if not prompt.strip():
        return (
            "⚠️ Please enter a question.",
            "",
            {lb: 0.0 for lb in OPTION_LABELS},
        )
    if not all(o.strip() for o in options):
        return (
            "⚠️ Please fill in all 5 options.",
            "",
            {lb: 0.0 for lb in OPTION_LABELS},
        )

    model, tokenizer, device = load_model()

    encoded = tokenizer(
        [prompt] * len(options),
        options,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    inputs = {k: v.unsqueeze(0).to(device) for k, v in encoded.items()}
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    ranked_idx = np.argsort(probs)[::-1]
    ranked_labels = [OPTION_LABELS[i] for i in ranked_idx]

    top3_str = " → ".join(ranked_labels[:3])
    best = ranked_labels[0]
    best_prob = probs[ranked_idx[0]] * 100

    prediction_str = (
        f"## ✅ Predicted Answer: **{best}**\n\n"
        f"**Confidence:** {best_prob:.1f}%\n\n"
        f"**Top-3 Ranking (MAP@3):** `{top3_str}`"
    )

    prob_dict = {OPTION_LABELS[i]: float(probs[i]) for i in range(5)}

    return prediction_str, top3_str, prob_dict


def load_example(evt: gr.SelectData, examples_state):
    ex = examples_state[evt.index]
    return ex["question"], ex["A"], ex["B"], ex["C"], ex["D"], ex["E"]


# ── CSS ───────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif !important; }

/* ── Page background ── */
.gradio-container {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #0f0c29 100%) !important;
    min-height: 100vh;
}

/* ── Hero banner ── */
#hero-banner {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 8px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.4);
}
#hero-banner h1 {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: white !important;
    margin: 0 0 6px 0 !important;
}
#hero-banner p {
    color: rgba(255,255,255,0.85) !important;
    font-size: 0.95rem !important;
    margin: 0 !important;
}

/* ── Section cards ── */
.section-card {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 14px !important;
    padding: 20px !important;
    backdrop-filter: blur(12px);
}

/* ── Labels ── */
label span, .gr-form label {
    color: #a5b4fc !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.03em !important;
}

/* ── Textboxes ── */
textarea, input[type="text"] {
    background: rgba(15, 12, 41, 0.7) !important;
    border: 1px solid rgba(99, 102, 241, 0.35) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
    outline: none !important;
}

/* ── Predict button ── */
#predict-btn {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 14px !important;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.5) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    letter-spacing: 0.04em !important;
}
#predict-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(99, 102, 241, 0.65) !important;
}
#predict-btn:active { transform: translateY(0) !important; }

/* ── Results markdown ── */
#result-box {
    background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(6,182,212,0.10)) !important;
    border: 1px solid rgba(16,185,129,0.3) !important;
    border-radius: 12px !important;
    padding: 18px 20px !important;
    color: #d1fae5 !important;
    font-size: 1rem !important;
    min-height: 90px;
}
#result-box h2 { color: #6ee7b7 !important; }

/* ── Confidence label widget ── */
.gr-label {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 12px !important;
}

/* ── Examples gallery ── */
#example-gallery .gallery-item {
    background: rgba(99, 102, 241, 0.08) !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 10px !important;
    color: #c7d2fe !important;
    font-weight: 500 !important;
    font-size: 0.83rem !important;
    transition: background 0.2s, border-color 0.2s, transform 0.15s !important;
    padding: 10px 14px !important;
    cursor: pointer !important;
}
#example-gallery .gallery-item:hover {
    background: rgba(99, 102, 241, 0.22) !important;
    border-color: #6366f1 !important;
    transform: translateY(-2px) !important;
}

/* ── Section headings ── */
.section-heading {
    color: #c7d2fe !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    margin-bottom: 10px !important;
}

/* ── Footer ── */
footer { display: none !important; }

/* ── Top-3 textbox ── */
#top3-box textarea {
    background: rgba(6,182,212,0.08) !important;
    border-color: rgba(6,182,212,0.3) !important;
    color: #67e8f9 !important;
    font-weight: 600 !important;
    text-align: center !important;
}
"""

# ── Hero HTML ─────────────────────────────────────────────────────────────────
HERO_HTML = """
<div id="hero-banner">
  <h1>🧠 Smart MCQ Solver</h1>
  <p>Powered by <strong>DeBERTa-v3</strong> fine-tuned on academic MCQ datasets &nbsp;·&nbsp;
     IIT Madras BS Data Science — DL &amp; GenAI Project (T2-2026)</p>
</div>
"""

# ── Build UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Smart MCQ Solver — DeBERTa-v3",
    css=CUSTOM_CSS,
) as demo:

    # Hidden state holding examples list
    examples_state = gr.State(EXAMPLES)

    # Hero
    gr.HTML(HERO_HTML)

    # ── Example picker ────────────────────────────────────────────────────────
    with gr.Group(elem_classes=["section-card"]):
        gr.Markdown("### 💡 Quick Examples — click any card to auto-fill", elem_classes=["section-heading"])
        example_gallery = gr.Dataset(
            components=["text"],
            samples=[[ex["label"]] for ex in EXAMPLES],
            label="",
            elem_id="example-gallery",
            samples_per_page=6,
        )

    gr.HTML("<div style='height:12px'></div>")

    # ── Main body ─────────────────────────────────────────────────────────────
    with gr.Row(equal_height=False):

        # Left — Input
        with gr.Column(scale=3, elem_classes=["section-card"]):
            gr.Markdown("### 📝 Question & Options", elem_classes=["section-heading"])

            prompt_input = gr.Textbox(
                label="Question / Prompt",
                placeholder="Type or paste your MCQ question here...",
                lines=3,
                elem_id="prompt",
            )

            with gr.Row():
                opt_a = gr.Textbox(label="Option A", placeholder="Option A", elem_id="opt_a")
                opt_b = gr.Textbox(label="Option B", placeholder="Option B", elem_id="opt_b")

            with gr.Row():
                opt_c = gr.Textbox(label="Option C", placeholder="Option C", elem_id="opt_c")
                opt_d = gr.Textbox(label="Option D", placeholder="Option D", elem_id="opt_d")

            opt_e = gr.Textbox(label="Option E", placeholder="Option E", elem_id="opt_e")

            predict_btn = gr.Button("🔍 Predict Answer", variant="primary", size="lg", elem_id="predict-btn")

        # Right — Results
        with gr.Column(scale=2, elem_classes=["section-card"]):
            gr.Markdown("### 📊 Model Output", elem_classes=["section-heading"])

            prediction_out = gr.Markdown(
                value="*Run the model to see results here.*",
                label="Prediction",
                elem_id="result-box",
            )

            top3_out = gr.Textbox(
                label="Top-3 Ranking  (MAP@3 order)",
                interactive=False,
                elem_id="top3-box",
            )

            prob_out = gr.Label(
                label="Confidence Scores — all 5 options",
                num_top_classes=5,
            )

            gr.Markdown(
                """
> **How to read:** The bar chart shows normalised probabilities across all 5 options.  
> A higher bar means the model is more confident about that choice.
""",
                elem_id="hint-text",
            )

    # ── Footer info ───────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="
        margin-top:24px;
        padding:18px 24px;
        background:rgba(255,255,255,0.03);
        border:1px solid rgba(255,255,255,0.08);
        border-radius:12px;
        color:#94a3b8;
        font-size:0.82rem;
        line-height:1.8;
        text-align:center;
    ">
        <strong style="color:#a5b4fc">Model:</strong> DeBERTa-v3-base — fine-tuned for 5-option MCQ &nbsp;|&nbsp;
        <strong style="color:#a5b4fc">Architecture:</strong> DebertaV2ForMultipleChoice &nbsp;|&nbsp;
        <strong style="color:#a5b4fc">Project:</strong> IIT Madras BS DS — DL &amp; GenAI T2-2026 &nbsp;|&nbsp;
        <strong style="color:#a5b4fc">Author:</strong> Shitanshu Chaurasiya · Roll No. 24F2006167
    </div>
    """)

    # ── Event wiring ──────────────────────────────────────────────────────────
    predict_btn.click(
        fn=predict,
        inputs=[prompt_input, opt_a, opt_b, opt_c, opt_d, opt_e],
        outputs=[prediction_out, top3_out, prob_out],
    )

    example_gallery.click(
        fn=load_example,
        inputs=[examples_state],
        outputs=[prompt_input, opt_a, opt_b, opt_c, opt_d, opt_e],
    )

if __name__ == "__main__":
    demo.launch()