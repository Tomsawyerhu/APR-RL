# 1. pip install -U huggingface_hub
# 2. export HF_ENDPOINT=https://hf-mirror.com
# 3. 下载数据集 huggingface-cli download --token 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 --repo-type dataset --resume-download openai/gsm8k --local-dir gsm8k
# 4. 下载模型 huggingface-cli download --token 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 --resume-download Qwen/Qwen3-4B --local-dir Qwen3-4B
# 5. ACCELERATE_LOG_LEVEL=info CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file zero3.yaml  grpo.py --config config_full.yaml (单卡，多卡指定--num_processes)
# 5. ACCELERATE_LOG_LEVEL=info CUDA_VISIBLE_DEVICES=1 accelerate launch grpo.py --config config_full.yaml (单卡，多卡指定--num_processes)
# 启动vllm在第一块显卡: CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model /root/autodl-tmp/apr-rl/Qwen2.5-Coder-3B-Instruct --enforce-eager true
# CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model /root/autodl-tmp/apr-rl/Qwen3-4B --enforce-eager false --host 0.0.0.0 --port 8000
# trl==0.18.2 vllm==0.9.2 torch==2.7.0
# export VLLM_ALLOW_INSECURE_SERIALIZATION=1
import logging
import os.path
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Optional
import ast
import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import get_last_checkpoint
import trl
from trl import GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config, get_quantization_config, \
    get_kbit_device_map

from execution import *

logger = logging.getLogger(__name__)
logger_file = './data/logfile.txt'
if os.path.exists(logger_file):
    os.remove(logger_file)

base_url = ""
api_key = ""
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'


def get_tokenizer(model_args: ModelConfig):
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    return tokenizer


def get_model(model_args: ModelConfig, training_args):
    """Get the model"""
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    return model


@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    system_prompt: Optional[str] = field(
        default=None, metadata={"help": "The optional system prompt to use for benchmarking."}
    )
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.


    Args:
        reward_funcs (`list[str]`):
            List of reward functions.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["format", "fixed_code_pass_all_test_reward",
                                 "testcase_pass_groundtruth_and_kill_bug_reward"],
        metadata={
            "help": "List of reward functions. Possible values: 'format', "
                    "'fixed_code_pass_all_test_reward', 'testcase_pass_groundtruth_and_kill_bug_reward'"
        }
    )

    train_dataset: str = field(
        default='', metadata={"help": "dataset used for training and testing"}
    )

    eval_dataset: Optional[str] = field(
        default="", metadata={"help": "Eval dataset"}
    )

    evaluate_only: bool = field(
        default=True, metadata={"help": "Do evaluation only"}
    )

    reasoning_model_enable_cot: bool = field(
        default=False, metadata={"help": "Enable think during train/eval, only effective for reasoning models"}
    )


def extract_python_code(text):
    text = text.split("```python")[-1]
    return text.split("```")[0]


def extract_testcases(text: str):
    text = text.split("```json")[-1]
    text = text.split('```')[0]
    try:
        # 尝试解析 JSON 字符串
        json_obj = json.loads(text)
        if not isinstance(json_obj, list):
            return None
        return json_obj
    except json.JSONDecodeError:
        return None


# 格式reward
# 测试用例reward: 语法正确性, 正确程序能够通过，bug程序不能通过
# 修复reward: 修复代码能通过所有用例

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


def testcase_pass_groundtruth_and_kill_bug_reward(completions, **kwargs):
    start_time = time.time()
    contents = [completion[0]["content"] for completion in completions]
    dataset_types = kwargs['dataset']
    rewards = []

    for i, content in enumerate(contents):
        problem = CodeRepairProblem(
            dataset=kwargs["dataset"][i], id=kwargs["id"][i],
            question=kwargs["question"][i],
            test_code=kwargs["test_code"][i],
            test_inputs=kwargs["test_inputs"][i], test_outputs=kwargs["test_outputs"][i],
            entry_point=kwargs["entry_point"][i], ground_truth=kwargs["ground_truth"][i],
            buggy_code=kwargs["buggy_code"][i]
        )

        if not check_format(content, dataset_types[i]):
            rewards.append(0)
            continue

        testcases = extract_testcases(content)
        logger.info("testcases are:\n%s", str(testcases))
        if testcases is None or not isinstance(testcases, list):
            rewards.append(0)
            continue

        pass_ground_truth = [0] * len(testcases)
        kill_bug = [0] * len(testcases)

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures_pass = []
            for idx, testcase in enumerate(testcases):
                futures_pass.append(executor.submit(
                    evaluate_generated_testcase,
                    testcase,
                    problem,
                    True,
                    dataset_types[i],
                    idx
                ))

            for future in as_completed(futures_pass):
                _idx, result = future.result()
                pass_ground_truth[_idx] = result

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures_kill = []
            for idx, testcase in enumerate(testcases):
                futures_kill.append(executor.submit(
                    evaluate_generated_testcase,
                    testcase,
                    problem,
                    False,
                    dataset_types[i],
                    idx
                ))

            for future in as_completed(futures_kill):
                _idx, result = future.result()
                kill_bug[_idx] = result

        logger.info("pass_ground_truth is %s", pass_ground_truth)
        logger.info("kill_bug is %s", kill_bug)
        testcase_validity = [x * y for x, y in zip(pass_ground_truth, kill_bug)]
        reward = (sum(testcase_validity) / len(testcase_validity)) if len(testcase_validity) > 0 else 0
        rewards.append(reward)

    end_time = time.time()
    logger.info("testcase_pass_groundtruth_and_kill_bug_reward cost %s seconds", end_time - start_time)
    logger.info("testcase_pass_groundtruth_and_kill_bug_reward is %s", rewards)
    with open(logger_file, 'a') as f:
        f.write(f'dataset = {dataset_types[0]}, testcase_pass_groundtruth_and_kill_bug_reward is {str(rewards)}\n')
    return rewards


def fixed_code_pass_all_test_reward(completions, **kwargs):
    start_time = time.time()
    contents = [completion[0]["content"] for completion in completions]
    dataset_types = kwargs['dataset']

    rewards = [0] * len(completions)
    futures = []

    with ThreadPoolExecutor(max_workers=16) as executor:
        for i, content in enumerate(contents):
            problem = CodeRepairProblem(
                dataset=kwargs["dataset"][i], id=kwargs["id"][i],
                question=kwargs["question"][i],
                test_code=kwargs["test_code"][i],
                test_inputs=kwargs["test_inputs"][i], test_outputs=kwargs["test_outputs"][i],
                entry_point=kwargs["entry_point"][i], ground_truth=kwargs["ground_truth"][i],
                buggy_code=kwargs["buggy_code"][i]
            )
            fixed_code = extract_python_code(content)

            if not is_valid_python(fixed_code):
                rewards[i] = 0
                continue

            futures.append(executor.submit(test_fixed_code, i, problem, fixed_code, dataset_types[i]))

        for future in as_completed(futures):
            idx, reward = future.result()
            rewards[idx] = reward  # 保持顺序

    end_time = time.time()
    logger.info("fixed_code_pass_all_test_reward cost %s seconds", end_time - start_time)
    logger.info("fixed_code_pass_all_test_reward is %s", rewards)
    with open(logger_file, 'a') as f:
        f.write(f'dataset = {dataset_types[0]}, fixed_code_pass_all_test_reward is {str(rewards)}\n')

    return rewards


def test_fixed_code(i, problem, fixed_code, dataset_type):
    logger.info("fixed_code is %s", fixed_code)
    exec_result = run_base_tests(problem, fixed_code)

    if dataset_type in [DatasetType.HUMAN_EVAL.value, DatasetType.CODE_FORCES.value, DatasetType.CODE_CONTESTS.value]:
        if isinstance(exec_result, dict):
            reward = exec_result.get('pass_rate', 0)
        else:
            reward = 0
    elif dataset_type == DatasetType.MBPP.value:
        if isinstance(exec_result, dict):
            reward = 1 if exec_result.get('passed', False) else 0
        else:
            reward = 0
    else:
        raise Exception(f'unsupported dataset type {dataset_type}')

    return i, reward


def check_format(completion, dataset_type):
    pattern = r"^.*?```json(.*?)```.*?```python(.*?)```.*?$"
    match_result = re.match(pattern, completion, flags=re.DOTALL)
    if completion.count("```json") != 1:
        return False
    if completion.count("```python") != 1:
        return False
    if not match_result:
        return False
    json_part = match_result.group(1).strip()
    python_code = match_result.group(2).strip()

    # Json格式是否正确,json object是否是list
    try:
        # 尝试解析 JSON 字符串
        json_obj = json.loads(json_part)
        if not isinstance(json_obj, list):
            return True
    except json.JSONDecodeError:
        return False
    # Python 代码语法是否正确
    if not is_valid_python(python_code):
        return False

    format_correct = True
    if dataset_type in [DatasetType.CODE_FORCES.value, DatasetType.CODE_CONTESTS.value]:
        # code forces 是test_input,test_output格式
        for item in json_obj:
            if not is_valid_test_input_output(item):
                format_correct = False
                break
    elif dataset_type in [DatasetType.HUMAN_EVAL.value, DatasetType.MBPP.value]:
        # humaneval和mbpp 是assert格式
        for item in json_obj:
            if not isinstance(item, str) or not is_valid_assertion(item):
                format_correct = False
                break
    else:
        raise Exception(f'unsupported dataset type {dataset_type}')
    return format_correct


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    dataset_types = kwargs['dataset']
    rewards = []

    completion_contents = [completion[0]["content"] for completion in completions]
    for i, completion in enumerate(completion_contents):
        dataset_type = dataset_types[i]
        if check_format(completion, dataset_type):
            rewards.append(1)
        else:
            rewards.append(0)

    logger.info("format_reward is %s", rewards)
    return rewards


def is_reasoning_model(model_name: str):
    if 'qwen3' in model_name.lower() or 'smollm3' in model_name.lower():
        return True
    return False


def think_reward(completions, **kwargs):
    pass


reward_funcs_registry = {
    "fixed_code_pass_all_test_reward": fixed_code_pass_all_test_reward,
    "testcase_pass_groundtruth_and_kill_bug_reward": testcase_pass_groundtruth_and_kill_bug_reward,
    "format": format_reward,
}

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


def main(script_args, training_args, model_args):
    def make_conversation(example, think_mode=None):
        problem = CodeRepairProblem.from_json(example)
        prompt_template = PROGRAM_REPAIR_TEMPLATE if problem.dataset in [DatasetType.CODE_FORCES.value,
                                                                         DatasetType.CODE_CONTESTS.value] else FUNCTION_REPAIR_TEMPLATE
        if think_mode is None:
            # 非推理模型
            messages = [
                {"role": "user", "content": prompt_template.format(faulty_function=problem.buggy_code)},
            ]
            logger.info('not a reasoning model')
        elif think_mode:
            # 推理模型开启CoT
            messages = [
                # {"role": "system", "content": "/think"},
                {"role": "user", "content": prompt_template.format(faulty_function=problem.buggy_code)+" /think"},
            ]
            logger.info('enable think mode')
        else:
            # 推理模型关闭CoT
            messages = [
                # {"role": "system", "content": "/no_think"},
                {"role": "user", "content": prompt_template.format(faulty_function=problem.buggy_code)+" /no_think"},
            ]
            logger.info('disable think mode')
        return {
            **example,
            "prompt": messages,
        }

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args)

    is_reasoning = is_reasoning_model(model_args.model_name_or_path)
    enable_think_mode = (script_args.reasoning_model_enable_cot if is_reasoning else None)
    logger.info(f"enable_think_mode={enable_think_mode}")

    # Load the dataset
    if 'jsonl' in script_args.train_dataset:
        train_dataset = load_dataset('json', data_files=script_args.train_dataset)
    else:
        train_dataset = load_dataset(script_args.train_dataset)
    # 打乱训练集
    train_dataset = train_dataset.shuffle()

    fn_kwargs = {"think_mode": enable_think_mode}

    if not training_args.do_eval:
        eval_dataset = None
    else:
        # Load the eval dataset
        if 'jsonl' in script_args.eval_dataset:
            eval_dataset = load_dataset('json', data_files=script_args.eval_dataset)
        else:
            eval_dataset = load_dataset(script_args.eval_dataset)
        eval_dataset = eval_dataset.map(make_conversation, fn_kwargs=fn_kwargs)

    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    train_dataset = train_dataset.map(make_conversation, fn_kwargs=fn_kwargs)
    logger.info("*** Initializing model ***")
    model = get_model(model_args, training_args)

    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset[script_args.dataset_train_split],
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    if not script_args.evaluate_only:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset[script_args.dataset_train_split])
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        ##################################
        # Save model and create model card
        ##################################
        logger.info("*** Save model ***")
        trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
        trainer.save_model(training_args.output_dir)
        logger.info(f"Model saved to {training_args.output_dir}")

        # Save everything else on main process
        kwargs = {
            "finetuned_from": model_args.model_name_or_path,
            "dataset": list(script_args.train_dataset),
            "dataset_tags": list(script_args.train_dataset),
            "tags": ["apr-rl"],
        }
        if trainer.accelerator.is_main_process:
            trainer.create_model_card(**kwargs)
            # Restore k,v cache for fast inference
            # trainer.model.config.use_cache = True
            trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
