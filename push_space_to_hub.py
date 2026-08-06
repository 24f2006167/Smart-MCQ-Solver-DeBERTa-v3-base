import os
import tempfile
from huggingface_hub import HfApi, create_repo, login

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

REQUIREMENTS = """torch>=2.1.0
transformers>=4.40.0
huggingface_hub>=0.34.0,<1.0
gradio>=5.0.0
numpy
sentencepiece
protobuf
audioop-lts
"""


def main():
    token = os.environ.get("HF_TOKEN") or input("HF Token: ").strip()
    login(token=token)

    api = HfApi()
    create_repo(SPACE_REPO_ID, repo_type="space", space_sdk="gradio", exist_ok=True, private=False)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tf:
        tf.write(SPACE_README)
        readme_tmp = tf.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tf:
        tf.write(REQUIREMENTS)
        req_tmp = tf.name

    api.upload_file(path_or_fileobj="app.py", path_in_repo="app.py", repo_id=SPACE_REPO_ID, repo_type="space")
    api.upload_file(path_or_fileobj=readme_tmp, path_in_repo="README.md", repo_id=SPACE_REPO_ID, repo_type="space")
    api.upload_file(path_or_fileobj=req_tmp, path_in_repo="requirements.txt", repo_id=SPACE_REPO_ID, repo_type="space")

    os.unlink(readme_tmp)
    os.unlink(req_tmp)

    print(f"Space updated: https://huggingface.co/spaces/{SPACE_REPO_ID}")


if __name__ == "__main__":
    main()