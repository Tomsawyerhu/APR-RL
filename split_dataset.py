import os
import jsonlines
import pandas as pd
from sklearn.model_selection import GroupKFold
from tqdm import tqdm


def read_jsonl(file_path):
    """读取 .jsonl 文件"""
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return pd.DataFrame(data)


def write_jsonl(data, file_path):
    """将 DataFrame 写入 .jsonl 文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with jsonlines.open(file_path, mode='w') as writer:
        for _, row in data.iterrows():
            writer.write(row.to_dict())


def split_dataset(split_list: list, output_dir: str, k=5):
    """
    对每个 JSONL 数据集单独进行 k 折划分，并保存训练/测试集。
    确保相同 id 的数据被划分到同一个 fold 中。

    参数：
    - merge_list: 要处理的 jsonl 文件路径列表
    - output_dir: 输出目录
    - k: 交叉验证的 fold 数
    """
    for dataset_file in tqdm(split_list, desc="Processing datasets"):
        # 提取文件名作为数据集名称
        base_name = os.path.basename(dataset_file).replace('.jsonl', '')

        # 读取数据集
        df = read_jsonl(dataset_file)

        # 随机打乱数据
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # 获取所有 id 列表（每个样本对应一个 id）
        groups = df['id'].values

        # 使用 GroupKFold 进行划分
        gkf = GroupKFold(n_splits=k)

        for fold, (train_idx, test_idx) in enumerate(
                tqdm(gkf.split(df, groups=groups), total=k, desc=f"Splitting {base_name}")):
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

            # 构造文件路径
            train_file = os.path.join(output_dir, f"{base_name}_train_fold_{fold + 1}.jsonl")
            test_file = os.path.join(output_dir, f"{base_name}_test_fold_{fold + 1}.jsonl")

            # 写入文件
            write_jsonl(train_df, train_file)
            write_jsonl(test_df, test_file)

            print(f"[{base_name}] Fold {fold + 1}:")
            print(f"  Train saved to: {train_file}")
            print(f"  Test saved to: {test_file}")


def merge_dataset(merge_list: list, output_file: str):
    """
    将多个 .jsonl 文件合并为一个 .jsonl 文件。

    参数：
    - merge_list: 要合并的 .jsonl 文件路径列表
    - output_file: 输出文件路径（.jsonl 格式）
    """
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with jsonlines.open(output_file, mode='w') as writer:
        for file_path in merge_list:
            print(f"Reading from {file_path}...")
            with jsonlines.open(file_path) as reader:
                for obj in reader:
                    writer.write(obj)

    print(f"✅ Merged dataset saved to: {output_file}")


if __name__ == '__main__':
    split_dataset(
        split_list=[
            './data/apr/codeforces_buggy_clean.jsonl',
            './data/apr/humaneval_buggy_clean.jsonl',
            './data/apr/mbpp_buggy_clean.jsonl'
        ],
        output_dir='./data/apr'
    )

    for k in range(1,6):
        merge_dataset(
            merge_list=[
                # f'./data/apr/codeforces_buggy_clean_train_fold_{k}.jsonl',
                f'./data/apr/humaneval_buggy_clean_train_fold_{k}.jsonl',
                f'./data/apr/mbpp_buggy_clean_train_fold_{k}.jsonl',
            ],
            output_file=f'./data/apr/apr_train_fold_{k}.jsonl',
        )
        merge_dataset(
            merge_list=[
                # f'./data/apr/codeforces_buggy_clean_test_fold_{k}.jsonl',
                f'./data/apr/humaneval_buggy_clean_test_fold_{k}.jsonl',
                f'./data/apr/mbpp_buggy_clean_test_fold_{k}.jsonl',
            ],
            output_file=f'./data/apr/apr_test_fold_{k}.jsonl',
        )


