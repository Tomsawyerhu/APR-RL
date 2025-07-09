import jsonlines
import outlines
from vllm import LLM, SamplingParams

from execution import *


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


def repair_humaneval():
    exec_cache = dict()
    with jsonlines.open('./data/humaneval_buggy_clean.jsonl', 'r') as f:
        for line in f:
            problem = CodeRepairProblem.from_json(line)
            repair_prompt = zero_shot_prompt(instruction=repair_instruction, question=problem.buggy_code)
            response = generate_by_llm(prompt=repair_prompt)
            fixed_function = extract_python_code(response)
            if fixed_function in exec_cache.keys():
                exec_result = exec_cache.get(fixed_function)
            else:
                exec_result = run_base_tests(problem, fixed_function)
                exec_cache[fixed_function] = exec_result
            line['fixed_code'] = fixed_function
            line['pass_rate'] = exec_result['pass_rate']
            with jsonlines.open('./data/humaneval_fix.jsonl', 'a') as f:
                f.write(line)


def repair_mbpp():
    exec_cache = dict()
    with jsonlines.open('./data/mbpp_buggy_clean.jsonl', 'r') as f:
        for line in f:
            problem = CodeRepairProblem.from_json(line)
            repair_prompt = zero_shot_prompt(instruction=repair_instruction, question=problem.buggy_code)
            response = generate_by_llm(prompt=repair_prompt)
            fixed_function = extract_python_code(response)
            if fixed_function in exec_cache.keys():
                exec_result = exec_cache.get(fixed_function)
            else:
                exec_result = run_base_tests(problem, fixed_function)
                exec_cache[fixed_function] = exec_result
            line['fixed_code'] = fixed_function
            line['passed'] = exec_result['passed']
            with jsonlines.open('./data/mbpp_fix.jsonl', 'a') as f:
                f.write(line)


def repair_both(repair_model_path='',repair_model_name=''):
    exec_cache = dict()
    total, passed = 0, 0
    if not os.path.exists('./data/result'):
        os.mkdir('./data/result')
    with jsonlines.open('./data/apr_test_fold_1.jsonl', 'r') as f:
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
            with jsonlines.open(f'./data/result/apr_test_fold_1_{repair_model_name}.jsonl', 'a') as ff:
                ff.write(line)
    print(f'pass@1 = {passed / total}')


if __name__ == '__main__':
    repair_both(repair_model_name='qwen2.5_coder_1.5b_instruct',repair_model_path='/root/autodl-tmp/apr-rl/Qwen2.5-Coder-1.5B-Instruct')
