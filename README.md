#### CECS327 Semester Project
**Group members**: Michael Bui, Daehee Kim

This repository holds various benchmarking tools to test the accuracy and resource utilization of two SLM models:
  - [unsloth\gemma-7b-it-bnb-4bit](https://huggingface.co/unsloth/gemma-7b-it-bnb-4bit)
  - [unsloth\llama-3-8b-Instruct-bnb-4bit](https://huggingface.co/unsloth/llama-3-8b-Instruct-bnb-4bit)

These models can be further assessed at your convenience within Huggingface's website directory.


### Accuracy Benchmarking (MMLU, GSM-8K, ARC_CHALLENGE)
To benchmark the accuracy of a LLM or SLM locally on your own machine, please follow the below instructions. These can be run from Powershell, Jupyter Notebook, or your IDE of choice:

```
# Virtual environment setup
python3 -m venv evals
source evals/bin/activate

# Install common libraries
pip install requests accelerate sentencepiece pytablewriter einops protobuf huggingface_hub==0.34
pip install -U transformers

# IMPORTANT -- to properly utilize GPU for benchmarking, compatible version of torch must be installed.
# Check your hardware specs to see which version of CUDA is needed.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

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
A demonstration of this benchmarking tool can be found via this link: [Evaluating Accuracy](https://www.youtube.com/watch?v=RtRN_BQRaG4)

------------------------------------------------------------------------------------------------------------------------------

### Evaluating Technical Performance
Note that our project was developed and run on Windows
machines. Python file is saved as "Semester Project.py"

1) Ensure that you have all of the requisite libraries
and dependencies.
  This includes, but is not limited to: pynvml,
  transformers, torch, accelerate, bitsandbytes,
  and IPython.
  These can be installed using the pip command.
  IMPORTANT: ensure that your torch version includes cuda
  support. You can check this with "pip show torch".

2) Open a command prompt and change directories to where
the project Python file is saved.

3) Run the file using the python command.

4) Input which model you wish to run along with the task
and whose machine is being used.

5) Wait for the SLM to generate its responses (might take
a while, especially if you have other programs running).

6) After the program finishes executing, you may view the
raw data collected in the .csv and .txt files the program
generated.

A demonstration of this evaluation can be found via this link: [Evaluating Technical Performance](https://www.youtube.com/watch?v=-1nIUqnzV6I
)
