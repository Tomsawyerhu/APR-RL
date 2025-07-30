import os

from huggingface_hub import HfApi, create_repo

os.environ['HF_TOKEN'] = ''
# 你的模型目录路径
model_paths = [
    "/root/autodl-tmp/apr-rl/Qwen2.5-3B-Instruct-APR-RL/checkpoint-5000",
    "/root/autodl-tmp/apr-rl/Qwen2.5-3B-Instruct-APR-RL/checkpoint-10000",
    "/root/autodl-tmp/apr-rl/Qwen2.5-3B-Instruct-APR-RL/checkpoint-11000",
    "/root/autodl-tmp/apr-rl/Qwen2.5-3B-Instruct-APR-RL-ABLATION-CODE/checkpoint-5000",
    "/root/autodl-tmp/apr-rl/Qwen2.5-3B-Instruct-APR-SFT/checkpoint-2711"
]
repo_ids = [
    "tomhu/Qwen2.5-Coder-3B-RL-5000-step",
    "tomhu/Qwen2.5-Coder-3B-RL-10000-step",
    "tomhu/Qwen2.5-Coder-3B-RL-11000-step",
    "tomhu/Qwen2.5-Coder-3B-RL-Ablation-Code-5000-step",
    "tomhu/Qwen2.5-Coder-3B-SFT",
]

for model_path, repo_id in zip(model_paths, repo_ids):
    # 如果还没有这个 repo，先创建它
    create_repo(repo_id, exist_ok=True, private=False)  # set private=True if you want a private repo

    # 推送本地目录到 Hugging Face
    api = HfApi()
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
    )
