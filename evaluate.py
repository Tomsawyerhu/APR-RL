import ast
import os.path

import jsonlines
import outlines
from tqdm import tqdm
from vllm import LLM, SamplingParams

from execution import *

model = None


def generate_by_llm(prompt, model_path="/mnt/data/hhc/Qwen2.5-Coder-3B-Instruct", kwargs=None):
    global model
    if kwargs is None:
        kwargs = {
            "tensor_parallel_size": 1,  # int(os.getenv("VLLM_N_GPUS", "1"))
            "dtype": "bfloat16",
            "trust_remote_code": True,
            "enforce_eager": True,
            "gpu_memory_utilization": 0.98
        }
    if model is None:
        model = LLM(model=model_path, max_model_len=4096, **kwargs)
    vllm_outputs = model.generate(
        prompts=[prompt],
        sampling_params=SamplingParams(
            temperature=1.0,
            max_tokens=2048,
            presence_penalty=1.0,  # 抑制新词出现频率过高
            frequency_penalty=1.0,  # 抑制已经出现过的词频繁重复
        ),
    )

    output_text = vllm_outputs[0].outputs[0].text.replace("\t", "    ")
    print(output_text)
    return output_text


FUNCTION_REPAIR_TEMPLATE = """
You are an expert in the field of software testing.
You are given a buggy Python function, then you are supposed to first generate assertions that can expose the defect,
and then generate the corresponding fixed code.
You may generate one or more assertions, return them in json list ```json```.
The fixed code should also be in the format ```python```.
Here is an example.
The faulty function is:
```python
def add (x, y):
    \"\"\"return sum of x and y\"\"\"
    return x - y
```

Assertions that can expose the bug:
```json
[
    \"assert add (1, 2) == 3\",
    \"assert add (-1, 1) == 0\",
    \"assert add (-1, 2) == 1\",
    \"assert add (10000, 1) == 10001\",
    \"assert add (-1, -2) == -3\"
]
```

Fixed code:
```python
def add (x, y):
    return x + y
```

Now, you are given a faulty Python function, please return:
1. **Assertions** that helps expose the bug.
2. **Fixed code** that can pass all testcases.

The faulty function is:
```python
{faulty_function}
```
"""
#
# FUNCTION_REPAIR_TEMPLATE = """
# You are an expert in the field of software testing.
# You are given a buggy Python function, you are supposed to first generate assertions that can expose the bug,
# and then generate the corresponding fixed code. The two tasks are detailed as follows.
#
# 1. **Generate a comprehensive set of assertions to expose the bug**:
#    - Each test case should be in the form of assertion.
#    - Write in ```json ``` block.
#
# 2. **Provide a fixed version**:
#    - Write a correct Python function to fix the bug.
#    - Write in ```python ``` block.
#
# The faulty function is:
# ```python
# {faulty_function}
# ```
#
# The **assertions and the fixed code** are:
# """

PROGRAM_REPAIR_TEMPLATE = """
You are an expert in the field of software testing. 
You are given a buggy Python program, you are supposed to first generate testcases that can expose the bug, 
and then generate the corresponding fixed code. The two tasks are detailed as follows.

1. **Generate a comprehensive set of test cases to expose the bug**:
   - Each test case should include an input and the expected output.
   - Output the test cases as a JSON list, where each entry is a dictionary with keys `"test_input"` and `"test_output"`.
   - Write in ```json ``` block.

2. **Provide a fixed version**:
   - Write a correct Python program to fix the bug.
   - Write in ```python ``` block.
   - The code should read from standard input and write to standard output, matching the input/output format specified in the problem.

Here is an example. 
The faulty Python program is:
```python
\"\"\"Please write a Python program to sum two integer inputs\"\"\"
def add (x, y):
    return x - y 
x = int(input())
y = int(input())
print(add(x,y))
```

Testcases that can expose the bug:
```json
[
    {{
        \"test_input\":\"1\n2\",
        \"test_output\":\"3\"
    }},
    {{
        \"test_input\":\"-1\n1\",
        \"test_output\":\"0\"
    }},
    {{
        \"test_input\":\"-1\n2\",
        \"test_output\":\"1\"
    }}
]
```

Fixed code:
```python
def add (x, y):
    return x + y 
x = int(input())
y = int(input())
print(add(x,y))
```

Now, you are given a faulty Python function, please return:
1. **Testcases** that helps expose the bug.
2. **Fixed code** that can pass all testcases.

The faulty function is:
```python
{faulty_function}
```
"""


def format_prompt(problem: CodeRepairProblem):
    if problem.dataset in [DatasetType.HUMAN_EVAL.value, DatasetType.MBPP.value]:
        return FUNCTION_REPAIR_TEMPLATE.format(faulty_function=problem.buggy_code)
    elif problem.dataset in [DatasetType.CODE_FORCES.value, DatasetType.CODE_CONTESTS.value]:
        return PROGRAM_REPAIR_TEMPLATE.format(faulty_function=problem.buggy_code)
    else:
        raise Exception('unsupported dataset')


# def fix_single_quotes(json_str):
#     """
#     将 JSON 字符串中的单引号替换为双引号，并尝试解析。
#     """
#     # 替换单引号为双引号
#     fixed_json_str = json_str.replace("'", '"')
#
#     # 添加额外修复：确保布尔值和 null 使用小写
#     fixed_json_str = fixed_json_str.replace("True", "true")
#     fixed_json_str = fixed_json_str.replace("False", "false")
#     fixed_json_str = fixed_json_str.replace("None", "null")
#
#     return fixed_json_str


def extract_testcases(response: str):
    response = response.split("```json")[-1]
    response = response.split('```')[0]
    try:
        # 尝试解析 JSON 字符串
        json_obj = json.loads(response)
        if not isinstance(json_obj, list):
            return None
        return json_obj
    except json.JSONDecodeError as e:
        print(e)
        return None


def extract_python_code(response):
    text = response
    if "```python" in text:
        text = text.split('```python')[-1]
    return text.split("```")[0]


def is_valid_python(code):
    if code is None:
        return False
    if not isinstance(code, str):
        return False
    if len(code.strip()) == 0:
        return False
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def is_valid_assertion(testcase):
    if is_valid_python(testcase):
        if 'assert' in testcase.lower():
            return True
        else:
            return False
    return False


def is_valid_test_input_output(testcase):
    if not isinstance(testcase, dict):
        return False
    if 'test_input' not in testcase.keys():
        return False
    if 'test_output' not in testcase.keys():
        return False
    return True


def evaluate_generated_testcase(testcase, problem, is_ground_truth, dataset_type, idx):
    """
    判断单个 testcase 是否 pass ground truth 或 kill bug.
    返回 (index, result)
    """
    if not is_valid_assertion(testcase) and dataset_type in [DatasetType.HUMAN_EVAL.value, DatasetType.MBPP.value]:
        return idx, 0  # 表示无效测试用例
    elif not is_valid_test_input_output(testcase) and dataset_type == DatasetType.CODE_FORCES.value:
        return idx, 0

    if is_ground_truth:
        exec_result = run_extra_tests(problem, problem.ground_truth, [testcase], check_on_gt=True)
        result = 1 if (isinstance(exec_result, dict) and exec_result.get('pass_rate', None) == 1) else 0
    else:
        exec_result = run_extra_tests(problem, problem.buggy_code, [testcase], check_on_gt=False)
        result = 1 if (isinstance(exec_result, dict) and exec_result.get('pass_rate', None) == 0) else 0
    return idx, result


def run_evaluation(evaluation_model_path='', input_file='', output_file=''):
    if not os.path.exists('./data/result'):
        os.mkdir('./data/result')
    exec_cache = dict()
    total_solution, passed_solution, discriminative_testcase_rate, discriminative_testcase_num = 0, 0, 0, 0
    total_lines = sum(1 for _ in jsonlines.open(input_file))
    with jsonlines.open(input_file, 'r') as f:
        for line in tqdm(f, total=total_lines, desc="Processing lines", unit="line"):
            total_solution += 1
            problem = CodeRepairProblem.from_json(line)
            # if problem.dataset!=DatasetType.HUMAN_EVAL.value:
            #     continue
            repair_prompt = format_prompt(problem)
            response = generate_by_llm(prompt=repair_prompt, model_path=evaluation_model_path)
            fixed_function = extract_python_code(response)
            test_cases = extract_testcases(response)

            # patch
            if fixed_function in exec_cache.keys():
                exec_result = exec_cache.get(fixed_function)
            else:
                exec_result = run_base_tests(problem, fixed_function)
                exec_cache[fixed_function] = exec_result
            line['fixed_code'] = fixed_function
            if problem.dataset in [DatasetType.HUMAN_EVAL.value, DatasetType.CODE_FORCES.value,
                                   DatasetType.CODE_CONTESTS.value]:
                if not isinstance(exec_result, dict) or 'pass_rate' not in exec_result.keys():
                    line['passed'] = False
                else:
                    line['passed'] = (exec_result['pass_rate'] == 1)
            elif DatasetType.MBPP.value == problem.dataset:
                if not isinstance(exec_result, dict) or 'passed' not in exec_result.keys():
                    line['passed'] = False
                else:
                    line['passed'] = exec_result['passed']
            else:
                raise Exception('unsupported dataset')
            if line['passed']:
                passed_solution += 1
                print('passed')
            else:
                print('failed')

            # testcase
            line['discriminative_test'] = test_cases if isinstance(test_cases, list) else []
            test_pass_rate = 0

            if isinstance(test_cases, list) and len(test_cases) > 0:
                pass_ground_truth = [0] * len(test_cases)
                kill_bug = [0] * len(test_cases)

                with ThreadPoolExecutor(max_workers=16) as executor:
                    futures_pass = []
                    for idx, testcase in enumerate(test_cases):
                        futures_pass.append(executor.submit(
                            evaluate_generated_testcase,
                            testcase,
                            problem,
                            True,
                            problem.dataset,
                            idx
                        ))

                    for future in as_completed(futures_pass):
                        _idx, result = future.result()
                        pass_ground_truth[_idx] = result

                with ThreadPoolExecutor(max_workers=16) as executor:
                    futures_kill = []
                    for idx, testcase in enumerate(test_cases):
                        futures_kill.append(executor.submit(
                            evaluate_generated_testcase,
                            testcase,
                            problem,
                            False,
                            problem.dataset,
                            idx
                        ))

                    for future in as_completed(futures_kill):
                        _idx, result = future.result()
                        kill_bug[_idx] = result

                testcase_validity = [x * y for x, y in zip(pass_ground_truth, kill_bug)]
                test_pass_rate = sum(testcase_validity) / len(testcase_validity) if len(testcase_validity) > 0 else 0

            # if isinstance(test_cases, list) and len(test_cases) > 0:
            #     test_result = run_extra_tests(problem, problem.ground_truth, test_cases, check_on_gt=True)
            #     if isinstance(test_result, dict) and 'pass_rate' in test_result.keys():
            #         test_pass_rate = test_result['pass_rate']
            line['discriminative_test_pass_rate'] = test_pass_rate
            discriminative_testcase_rate += test_pass_rate
            discriminative_testcase_num += len(test_cases) if isinstance(test_cases, list) else 0

            with jsonlines.open(output_file, 'a') as ff:
                ff.write(line)
    print(
        f'pass@1 = {passed_solution / total_solution}, test@1={discriminative_testcase_rate / total_solution}, generate {discriminative_testcase_num} tests in all')


if __name__ == '__main__':
    run_evaluation(evaluation_model_path='/root/autodl-tmp/apr-rl/Qwen2.5-1.5B-Instruct-APR-RL/checkpoint-3809',
                   input_file='./data/apr_test_fold_1.jsonl',
                   output_file='./data/result/apr_test_fold_1_rl.jsonl')
    # run_evaluation(evaluation_model_path='/root/autodl-tmp/apr-rl/Qwen2.5-1.5B-Instruct-APR-SFT/checkpoint-1473',
    #                input_file='./data/apr_test_fold_1.jsonl',
    #                output_file='./data/result/apr_test_fold_1_qwen2.5_coder_1.5b_instruct_sft_both.jsonl')
