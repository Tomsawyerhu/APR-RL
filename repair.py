import os

import jsonlines
import outlines
from vllm import LLM, SamplingParams

from execution import *

os.environ['CUDA_VISIBLE_DEVICES']='1'

@outlines.prompt
def zero_shot_prompt(instruction, question):
    """
    <|im_start|>system
    {{ instruction }}
    <|im_end|>

    <|im_start|>user
    The following Python function contains a bug. Can you help me fix it?

    Question:
    ```python
    {{ question }}
    ```
    <|im_end|>

    <|im_start|>assistant
    Answer:
    ```python
    """


repair_instruction = "You are an intelligent programming assistant to help fix bugs in Python programs."
STOP_SEQUENCES = ["```"]
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
            stop=STOP_SEQUENCES,
        ),
    )
    output_text = vllm_outputs[0].outputs[0].text.replace("\t", "    ")
    return output_text


def extract_python_code(response):
    text = response
    if "```python" in text:
        text = text.split('```python')[-1]
    return text.split("```")[0]


def repair(repair_model_path='',input_file='',output_file=''):
    exec_cache = dict()
    total, passed = 0, 0
    if not os.path.exists('./data/result'):
        os.mkdir('./data/result')
    with jsonlines.open(input_file, 'r') as f:
        for line in f:
            total += 1
            problem = CodeRepairProblem.from_json(line)
            repair_prompt = zero_shot_prompt(instruction=repair_instruction, question=problem.buggy_code)
            response = generate_by_llm(prompt=repair_prompt, model_path=repair_model_path)
            fixed_function = extract_python_code(response)
            if fixed_function in exec_cache.keys():
                exec_result = exec_cache.get(fixed_function)
            else:
                exec_result = run_base_tests(problem, fixed_function)
                exec_cache[fixed_function] = exec_result
            line['fixed_code'] = fixed_function
            if problem.dataset in [DatasetType.HUMAN_EVAL.value, DatasetType.CODE_FORCES.value,DatasetType.CODE_CONTESTS.value]:
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
                passed += 1
                print('passed')
            else:
                print('failed')
            with jsonlines.open(output_file, 'a') as ff:
                ff.write(line)
    print(f'pass@1 = {passed / total}')


if __name__ == '__main__':
    repair(repair_model_path='/root/autodl-tmp/apr-rl/Qwen2.5-1.5B-Instruct-APR-SFT/checkpoint-3809',
                input_file='./data/apr_test_fold_1.jsonl',
                output_file='./data/result/apr_test_fold_1_qwen2.5_coder_1.5b_instruct_sft_code.jsonl')

