import os
import tempfile
from huggingface_hub import HfApi

HF_USERNAME = "Shitanshu06"
SPACE_REPO_ID = f"{HF_USERNAME}/smart-mcq-solver"

SPACE_README = """---
title: Smart MCQ Solver
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "5.34.0"
app_file: app.py
pinned: false
license: apache-2.0
---

# Smart MCQ Solver — DeBERTa-v3

Fine-tuned DeBERTa-v3-base for 5-option MCQ answering.
"""

REQUIREMENTS = """torch
transformers>=4.40.0
sentencepiece
protobuf
numpy
"""


def main():
    api = HfApi()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tf:
        tf.write(SPACE_README)
        readme_tmp = tf.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tf:
        tf.write(REQUIREMENTS)
        req_tmp = tf.name

    print(f"Uploading files to Hugging Face Space: {SPACE_REPO_ID}...")
    api.upload_file(path_or_fileobj="app.py", path_in_repo="app.py", repo_id=SPACE_REPO_ID, repo_type="space")
    api.upload_file(path_or_fileobj="inference.py", path_in_repo="inference.py", repo_id=SPACE_REPO_ID, repo_type="space")
    api.upload_file(path_or_fileobj=readme_tmp, path_in_repo="README.md", repo_id=SPACE_REPO_ID, repo_type="space")
    api.upload_file(path_or_fileobj=req_tmp, path_in_repo="requirements.txt", repo_id=SPACE_REPO_ID, repo_type="space")

    os.unlink(readme_tmp)
    os.unlink(req_tmp)

    print("Restarting Space...")
    api.restart_space(repo_id=SPACE_REPO_ID)
    print(f"Space updated successfully: https://huggingface.co/spaces/{SPACE_REPO_ID}")


if __name__ == "__main__":
    main()