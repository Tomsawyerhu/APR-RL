# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ACCELERATE_LOG_LEVEL=info CUDA_VISIBLE_DEVICES=0,1 accelerate launch  sft.py --config sft_config.yaml

"""
# Full training
```
python trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eos_token '<|im_end|>' \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir Qwen2-0.5B-SFT \
    --push_to_hub
```

# LoRA
```
python trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-4 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eos_token '<|im_end|>' \
    --eval_strategy steps \
    --eval_steps 100 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --output_dir Qwen2-0.5B-SFT \
    --push_to_hub
```
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config, DataCollatorForCompletionOnlyLM,
)

from model import CodeRepairProblem


# os.environ['CUDA_VISIBLE_DEVICES']='1'
logger = logging.getLogger(__name__)
prompt_format = 'The following Python code contains bugs, please fix it.\n{buggy_code}'
response_format = 'The fixed code is:\n```python\n{fixed_code}\n```'


@dataclass
class SFTScriptArguments(ScriptArguments):
    """
    Script arguments for the SFT training script.
    """

    train_dataset: str = field(
        default='', metadata={"help": "dataset used for training and testing"}
    )

    eval_dataset: Optional[str] = field(
        default="", metadata={"help": "Eval dataset"}
    )


def main(script_args, training_args, model_args):
    def make_conversation(example):
        problem = CodeRepairProblem.from_json(example)

        return {
            "prompt": prompt_format.format(buggy_code=problem.buggy_code),
            "completion": response_format.format(fixed_code=problem.ground_truth)
        }

    # def formatting_prompts_func(example):
    #     output_texts = []
    #     for i in range(len(example['buggy_code'])):
    #         text = f"### INSTRUCTION\nThe following Python code contains bugs, please fix it.\n {example['buggy_code'][i]}\n ### Answer\n{example['ground_truth'][i]}"
    #         output_texts.append(text)
    #     return output_texts

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
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # Create model
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )

    ################
    # Dataset
    ################
    # Load the dataset
    if 'jsonl' in script_args.train_dataset:
        train_dataset = load_dataset('json', data_files=script_args.train_dataset)
    else:
        train_dataset = load_dataset(script_args.train_dataset)
    train_dataset = train_dataset.map(make_conversation)
    # 打乱训练集
    train_dataset = train_dataset.shuffle()

    if not training_args.do_eval:
        eval_dataset = None
    else:
        # Load the eval dataset
        if 'jsonl' in script_args.eval_dataset:
            eval_dataset = load_dataset('json', data_files=script_args.eval_dataset)
        else:
            eval_dataset = load_dataset(script_args.eval_dataset)
        eval_dataset = eval_dataset.map(make_conversation)

    ################
    # Training
    ################
    # response_template = " ### Answer"
    # collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    training_args.max_length = 2048
    training_args.completion_only_loss = True
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset[script_args.dataset_train_split],
        eval_dataset=eval_dataset,
        # formatting_func=formatting_prompts_func,
        # data_collator=collator,
        peft_config=get_peft_config(model_args),
    )

    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.train_dataset)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        # metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args, _ = parser.parse_args_and_config(return_remaining_strings=True)
    main(script_args, training_args, model_args)
