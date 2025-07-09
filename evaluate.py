import os.path

import jsonlines
import outlines
from vllm import LLM, SamplingParams

from execution import *
model = None


def generate_by_llm(prompt, model_path="/mnt/data/hhc/Qwen2.5-Coder-3B-Instruct", kwargs=None):
    global model
    if kwargs is None:
        kwargs = {
            "tensor_parallel_size": 1,  # int(os.getenv("VLLM_N_GPUS", "1"))
            "dtype": "float16",
            "trust_remote_code": True,
            "enforce_eager": True,
            "gpu_memory_utilization": 0.98
        }
    if model is None:
        model = LLM(model=model_path, max_model_len=1536, **kwargs)
    vllm_outputs = model.generate(
        prompts=[prompt],
        sampling_params=SamplingParams(
            temperature=0.0,
            max_tokens=2048,
        ),
    )
    output_text = vllm_outputs[0].outputs[0].text.replace("\t", "    ")
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
    'assert add (1, 2) == 3',
    'assert add (-1, 1) == 0',
    'assert add (-1, 2) == 1',
    'assert add (10000, 1) == 10001',
    'assert add (-1, -2) == -3'
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
        'test_input':'1\n2',
        'test_output':'3'
    }},
    {{
        'test_input':'-1\n1',
        'test_output':'0'
    }},
    {{
        'test_input':'-1\n2',
        'test_output':'1'
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
    if problem.dataset == DatasetType.HUMAN_EVAL.value or problem.dataset == DatasetType.MBPP.value:
        return FUNCTION_REPAIR_TEMPLATE.format(faulty_function=problem.buggy_code)
    else:
        raise Exception('unsupported dataset')


def extract_python_code(response):
    text = response
    if "```python" in text:
        text = text.split('```python')[-1]
    return text.split("```")[0]


def run_evaluation(evaluation_model_path=''):
    if not os.path.exists('./data/result'):
        os.mkdir('./data/result')
    exec_cache = dict()
    total, passed = 0, 0
    with jsonlines.open('./data/apr_test_fold_1.jsonl', 'r') as f:
        for line in f:
            total += 1
            problem = CodeRepairProblem.from_json(line)
            repair_prompt = format_prompt(problem)
            response = generate_by_llm(prompt=repair_prompt, model_path=evaluation_model_path)
            fixed_function = extract_python_code(response)
            if fixed_function in exec_cache.keys():
                exec_result = exec_cache.get(fixed_function)
            else:
                exec_result = run_base_tests(problem, fixed_function)
                exec_cache[fixed_function] = exec_result
            line['fixed_code'] = fixed_function
            if DatasetType.HUMAN_EVAL.value == problem.dataset:
                line['passed'] = (exec_result['pass_rate'] == 1)
            elif DatasetType.MBPP.value == problem.dataset:
                line['passed'] = exec_result['passed']
            else:
                raise Exception('unsupported dataset')
            if line['passed']:
                passed += 1
                print('passed')
            else:
                print('failed')
            with jsonlines.open(f'./data/result/apr_test_fold_1_eval.jsonl', 'a') as f:
                f.write(line)
    print(f'pass@1 = {passed / total}')


if __name__ == '__main__':
    run_evaluation(evaluation_model_path='/root/autodl-tmp/apr-rl/Qwen2.5-1.5B-Instruct-APR-RL/checkpoint-1473')
