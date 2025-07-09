import dataclasses

import jsonlines
from datasets import load_dataset
from openai import OpenAI
import pyarrow.parquet as pq
import re

from tqdm import tqdm

from execution import run_base_tests
from model import DatasetType, CodeRepairProblem

base_url = "https://api5.xhub.chat/v1"
api_key = "sk-KxyaIwYE6GyqdubcSNbcmDa4s4poc1okauEFzJNL4RiGaca8"


def make_model():
    _client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    return _client


gpt_client = make_model()


def generate(prompt, model_name="gpt-4o-mini", timeout=100, retry=100, num_samples=1):
    temperature = 0.0 if num_samples == 1 else 0.4
    if isinstance(prompt, str):
        prompt = [{
            'role': 'user',
            'content': prompt
        }]
    for i in range(retry):
        completion = gpt_client.chat.completions.create(
            model=model_name,
            messages=prompt,
            max_tokens=8000,
            temperature=temperature,
            n=num_samples,
            timeout=timeout
        )
        if not completion.choices or len(completion.choices) == 0:
            continue
        else:
            texts = [x.message.content for x in completion.choices]
            return texts
    print("No reply from GPT")
    return ""


def extract_python_code(response):
    text = response
    if "```python" in text:
        text = text.split('```python')[-1]
    return text.split("```")[0]


prompt = """You are an experienced software developer with a deep understanding of common human mistakes and subtle 
logic errors. Your task is to take a correct, well-written Python function as input and **introduce one or more 
realistic runtime bugs** that:

1. **Cause the program to fail at runtime** (e.g., `IndexError`, `KeyError`, `ZeroDivisionError`, etc.).
2. **Are logically misleading** — they should appear correct at first glance but contain subtle flaws.
3. **Mimic how a real programmer might make the mistake**, not just random syntax errors.
4. **Preserve the original structure and intent** of the function as much as possible.
5. **Take border situation into consideration.** The injected bugs should be as subtle as possible, especially targeting edge cases.
6. If the input contains test cases, the injected bugs should avoid triggering them or passing through them undetected.
7. **Do not change the function signature or return type**.

Input:
- The original correct code

Output:
- The buggy code after modification

Here is an example. 
Input:

[Start of Correct Code]
```python
from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False
```
[End of Correct Code]


Output:

[Start of Buggy Code]
```python
from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx < idx2:  # ❌ Bug: changed from `idx != idx2` to `idx < idx2`
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False
```
[End of Buggy Code]

Now, you are given a correct Python code and you should return a buggy version.
Input:
[Start of Correct Code]
```python
{correct_function}
```
[End of Correct Code]

"""


def construct_humaneval_buggy_dataset():
    # 打开 parquet 文件
    table = pq.read_table("./data/humaneval.parquet")

    # 转换为 DataFrame 并遍历
    df = table.to_pandas()

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing HumanEval dataset"):
        python_func = row['prompt'] + row['canonical_solution']
        buggy_funcs = generate(prompt.format(correct_function=python_func), num_samples=10)
        buggy_funcs = [extract_python_code(x) for x in buggy_funcs]

        # 将 row 转换为 dict，并添加新的字段 'buggy'
        row_dict = row.to_dict()
        row_dict['buggy'] = buggy_funcs

        # 使用追加模式 ('a') 写入 JSONL 文件
        with jsonlines.open('./data/humaneval_buggy.jsonl', 'a') as f:
            f.write(row_dict)


def construct_mbpp_buggy_dataset():
    # 打开 parquet 文件
    table = pq.read_table("./data/mbpp.parquet")

    # 转换为 DataFrame 并遍历
    df = table.to_pandas()

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing MBPP dataset"):
        python_func = "\"\"\"" + row['prompt'] + "\"\"\"\n\n" + row['code']
        buggy_funcs = generate(prompt.format(correct_function=python_func), num_samples=10)
        buggy_funcs = [extract_python_code(x) for x in buggy_funcs]

        # 将 row 转换为 dict，并添加新的字段 'buggy'
        row_dict = row.to_dict()
        row_dict['test_imports'] = list(row_dict['test_imports'])
        row_dict['test_list'] = list(row_dict['test_list'])
        row_dict['buggy'] = buggy_funcs

        # 使用追加模式 ('a') 写入 JSONL 文件
        with jsonlines.open('./data/mbpp_buggy.jsonl', 'a') as f:
            f.write(row_dict)


def construct_code_forces_buggy_dataset():
    # 打开 parquet 文件
    codeforces = load_dataset('json', data_files="./data/codeforces.jsonl")['train']

    for row in tqdm(codeforces, total=len(codeforces), desc="Processing Codeforces"):
        python_func = "\"\"\"" + row['orig_prompt'] + "\"\"\"\n\n" + row['ground_truth']
        buggy_funcs = generate(prompt.format(correct_function=python_func), num_samples=10)
        buggy_funcs = [extract_python_code(x) for x in buggy_funcs]

        # 将 row 转换为 dict，并添加新的字段 'buggy'
        row_dict = row
        row_dict['buggy'] = buggy_funcs

        # 使用追加模式 ('a') 写入 JSONL 文件
        with jsonlines.open('./data/codeforces_buggy.jsonl', 'a') as f:
            f.write(row_dict)


def remove_inline_comments(code: str) -> str:
    return re.sub(r'#.*$', '', code, flags=re.MULTILINE)


def filter_humaneval_buggy_dataset():
    buggy_programs = set()
    with jsonlines.open('./data/humaneval_buggy.jsonl', 'r') as f:
        for item in f:
            for buggy_program in item['buggy']:
                buggy_program = remove_inline_comments(buggy_program)
                if buggy_program in buggy_programs:
                    continue
                buggy_programs.add(buggy_program)
                problem = CodeRepairProblem(
                    dataset=DatasetType.HUMAN_EVAL.value,
                    id=item['task_id'],
                    question=item['prompt'],
                    test_code=item['test'],
                    test_inputs=[],
                    test_outputs=[],
                    entry_point=item['entry_point'],
                    ground_truth=item['prompt']+'\n'+item['canonical_solution'],
                    buggy_code=buggy_program
                )
                result = run_base_tests(problem, problem.buggy_code)
                if int(result['pass_rate']) == 1:
                    continue

                with jsonlines.open('./data/humaneval_buggy_clean.jsonl', 'a') as f:
                    f.write(dataclasses.asdict(problem))


def filter_mbpp_buggy_dataset():
    buggy_programs = set()
    with jsonlines.open('./data/mbpp_buggy.jsonl', 'r') as f:
        for item in f:
            for buggy_program in item['buggy']:
                buggy_program = remove_inline_comments(buggy_program)
                if buggy_program in buggy_programs:
                    continue
                buggy_programs.add(buggy_program)

                problem = CodeRepairProblem(
                    dataset=DatasetType.MBPP.value,
                    id=f"MBPP/{item['task_id']}",
                    question=item['prompt'],
                    test_code=item['test'],
                    test_inputs=[],
                    test_outputs=[],
                    entry_point='',
                    ground_truth=item['code'],
                    buggy_code=buggy_program
                )
                result = run_base_tests(problem, problem.buggy_code)
                if result['passed']:
                    continue

                with jsonlines.open('./data/mbpp_buggy_clean.jsonl', 'a') as f:
                    f.write(dataclasses.asdict(problem))


def filter_code_forces_buggy_dataset():
    buggy_programs = set()
    with jsonlines.open('./data/codeforces_buggy.jsonl', 'r') as f:
        for item in f:
            for buggy_program in item['buggy']:
                buggy_program = remove_inline_comments(buggy_program)
                if buggy_program in buggy_programs:
                    continue
                buggy_programs.add(buggy_program)

                problem = CodeRepairProblem(
                    dataset=DatasetType.CODE_FORCES.value,
                    id=item['id'],
                    question=item['orig_prompt'],
                    test_code='',
                    test_inputs=item['test_inputs'],
                    test_outputs=item['test_outputs'],
                    entry_point='',
                    ground_truth=item['ground_truth'],
                    buggy_code=buggy_program
                )
                result = run_base_tests(problem, problem.buggy_code)
                if int(result['pass_rate']) == 1:
                    continue

                with jsonlines.open('./data/codeforces_buggy_clean.jsonl', 'a') as f:
                    f.write(dataclasses.asdict(problem))


if __name__ == '__main__':
    # construct_humaneval_buggy_dataset()
    # construct_mbpp_buggy_dataset()
    # construct_code_forces_buggy_dataset()
    filter_humaneval_buggy_dataset()
    filter_mbpp_buggy_dataset()
