# APR-RL
GRPO for Automated Program Repair

## Run
#### GRPO

```shell
 CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model /root/autodl-tmp/apr-rl/Qwen2.5-Coder-3B-Instruct --enforce-eager true

 ACCELERATE_LOG_LEVEL=info CUDA_VISIBLE_DEVICES=1 accelerate launch grpo.py --config config_full.yaml
 ```


#### SFT

```shell
ACCELERATE_LOG_LEVEL=info CUDA_VISIBLE_DEVICES=0,1 accelerate launch  sft.py --config sft_config.yaml
```
