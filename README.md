#### CECS327 Semester Project
**Group members**: Michael Bui, Daehee Kim

This repository holds various benchmarking tools to test the accuracy and resource utilization of two SLM models:
  - [unsloth\gemma-7b-it-bnb-4bit](https://huggingface.co/unsloth/gemma-7b-it-bnb-4bit)
  - [unsloth\llama-3-8b-Instruct-bnb-4bit](https://huggingface.co/unsloth/llama-3-8b-Instruct-bnb-4bit)

These models can be further assessed at your convenience within Huggingface's website directory.


##### Accuracy Benchmarking (MMLU, GSM-8K, ARC_CHALLENGE)
To benchmark the accuracy of a LLM or SLM locally on your own machine, please follow the following commands on your Powershell directory:

```
# Virtual environment setup
python3 -m venv evals
source evals/bin/activate

# Install common libraries
pip install requests accelerate sentencepiece pytablewriter einops protobuf huggingface_hub==0.21.4
pip install -U transformers

# Accelerate Config
accelerate config default

# Install LLM Eval Harness
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .

# HF Login
huggingface-cli login 

# Arc-C 25 shot Example
accelerate launch -m lm_eval \
    --model hf \
    # Place accessible huggingface model after pretrained=... as needed
    --model_args pretrained=unsloth/gemma-7b-it-bnb-4bit,trust_remote_code=True,dtype=auto \
    # Tasks interchangeable (ex. arc_challenge, mmlu, gsm8k)
    --tasks arc_challenge \
    # Number of examples fed to model before test begins (0, 5, 10...)
    --num_fewshot 0 \
    --device cuda:0 \
    --batch_size 2 \
    # Configure output of .json file name here
    --output_path ./arcc_0_gemma.json
```
