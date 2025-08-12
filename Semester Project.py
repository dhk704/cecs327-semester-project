# ==================================================
# Group members: Michael Bui, Daehee Kim
# Student IDs: 14704389, 033241115
# Due date: 8/12/2025
# ==================================================
import time
import pynvml
import transformers
import torch
from IPython.display import clear_output
import csv
import psutil
import threading
from enum import Enum

class ModelChosen(Enum):
    LLAMA = 0
    GEMMA = 1

class TaskChosen(Enum):
    CREATIVE_WRITING = 0
    SUMMARIZING = 1

class MachineChosen(Enum):
    DAVID = 0
    MICHAEL = 1

def monitor_usage_to_csv(which_model, which_task, whose_machine):
    # Interval represents how often a data point is recorded (in seconds)
    interval = 3

    # Default filename
    filename = "usage_log.csv"

    # Initialize GPU monitoring
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    # Get current process
    process = psutil.Process()

    # Determine filename
    # Model
    if which_model == ModelChosen.LLAMA.value:
        filename = "llama"
    else:
        filename = "gemma"

    # Task
    if which_task == TaskChosen.CREATIVE_WRITING.value:
        filename += "_creative_writing"
    else:
        filename += "_summarization"

    # Whose machine used
    if whose_machine == MachineChosen.DAVID.value:
        filename += "_david"
    else:
        filename += "_michael"

    # Append .csv to filename
    filename += ".csv"

    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Time", "CPU Util (%)", "GPU Util (%)", "GPU Mem Util (%)", "GPU Mem Used (MB)", "GPU Mem Total (MB)"])
        time_data_recorded = 0

        while True:
            # CPU util for this process
            # Divide by number of cores to get overall CPU usage; else, it reports how many cores are saturated ie. 200% would mean two cores are fully used
            cpu_util = process.cpu_percent(interval=None) / psutil.cpu_count()
            gpu_util_info = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)

            # Get GPU stats
            gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            gpu_mem_used = gpu_mem_info.used / (1024 ** 2)
            gpu_mem_total = gpu_mem_info.total / (1024 ** 2)
            gpu_util = gpu_util_info.gpu
            gpu_mem_util = gpu_util_info.memory

            # Save to CSV
            writer.writerow([
                time_data_recorded,
                cpu_util,
                gpu_util,
                gpu_mem_util,
                gpu_mem_used,
                gpu_mem_total
            ])
            file.flush()
            time_data_recorded += interval

            # Sleep for interval to record data every interval seconds
            time.sleep(interval)

def generate_pipeline(model, task="text-generation"):
  pipeline = transformers.pipeline(
    # Parameters to pass to the model
    task,
    model=model,
    model_kwargs={
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
    })
  return pipeline

def chat_with_model(pipeline, user_input, max_new_tokens=512):
    global conversation_history

    # Append user message to history
    conversation_history.append({"role": "user", "content": user_input})

    # Generate prompt with full history
    prompt = pipeline.tokenizer.apply_chat_template(
        conversation_history,
        tokenize=False,
        add_generation_prompt=True
    )

    # Generate response
    outputs = pipeline(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )

    # Extract and store assistant's reply
    assistant_reply = outputs[0]["generated_text"][len(prompt):]
    conversation_history.append({"role": "assistant", "content": assistant_reply})

    # Show only current prompt and answer
    print(f"You: {conversation_history[-2]['content']}\n")
    print(f"Assistant: {conversation_history[-1]['content']}\n")

    return



# Set to avoid warning messages
transformers.logging.set_verbosity_error()

conversation_history = []

# Test models to see results
model_llama = "unsloth/llama-3-8b-Instruct-bnb-4bit"
model_gemma = "unsloth/gemma-7b-it-bnb-4bit"

# Question banks
creative_writing_questions_bank = [
    "Write a creative and interesting short story set in medieval times. Keep the story within 1,800 to 2,000 characters in length.",
    "Write a creative and interesting short story about the life of plants. Keep the story within 1,800 to 2,000 characters in length.",
    "Write a creative and interesting short story about a cat and a dog going on an adventure. Keep the story within 1,800 to 2,000 characters in length."
]
summarizing_questions_bank = [
    "Summarize the Odyssey by Homer in less than 2,000 characters.",
    "Summarize the following passage in less than 2,000 characters.\n"
    "The earliest molecular changes in Alzheimer's disease (AD) are poorly understood1-5. Here we show that endogenous lithium (Li) is dynamically regulated in the brain and contributes to cognitive preservation during ageing. Of the metals we analysed, Li was the only one that was significantly reduced in the brain in individuals with mild cognitive impairment (MCI), a precursor to AD. Li bioavailability was further reduced in AD by amyloid sequestration. We explored the role of endogenous Li in the brain by depleting it from the diet of wild-type and AD mouse models. Reducing endogenous cortical Li by approximately 50% markedly increased the deposition of amyloid-β and the accumulation of phospho-tau, and led to pro-inflammatory microglial activation, the loss of synapses, axons and myelin, and accelerated cognitive decline. These effects were mediated, at least in part, through activation of the kinase GSK3β. Single-nucleus RNA-seq showed that Li deficiency gives rise to transcriptome changes in multiple brain cell types that overlap with transcriptome changes in AD. Replacement therapy with lithium orotate, which is a Li salt with reduced amyloid binding, prevents pathological changes and memory loss in AD mouse models and ageing wild-type mice. These findings reveal physiological effects of endogenous Li in the brain and indicate that disruption of Li homeostasis may be an early event in the pathogenesis of AD. Li replacement with amyloid-evading salts is a potential approach to the prevention and treatment of AD.",
    "Summarize the following passage in less than 2,000 characters.\n"
    "The theory of relativity usually encompasses two interrelated physics theories by Albert Einstein: special relativity and general relativity, proposed and published in 1905 and 1915, respectively. Special relativity applies to all physical phenomena in the absence of gravity. General relativity explains the law of gravitation and its relation to the forces of nature.[2] It applies to the cosmological and astrophysical realm, including astronomy. The theory transformed theoretical physics and astronomy during the 20th century, superseding a 200-year-old theory of mechanics created primarily by Isaac Newton. It introduced concepts including 4-dimensional spacetime as a unified entity of space and time, relativity of simultaneity, kinematic and gravitational time dilation, and length contraction. In the field of physics, relativity improved the science of elementary particles and their fundamental interactions, along with ushering in the nuclear age. With relativity, cosmology and astrophysics predicted extraordinary astronomical phenomena such as neutron stars, black holes, and gravitational waves."
]

# Get user choice of what model and tests to run
which_model = input("\nSelect a model to run (0 for Llama, 1 for Gemma): ")
while (which_model != str(ModelChosen.LLAMA.value)) and (which_model != str(ModelChosen.GEMMA.value)):
    which_model = input("INVALID INPUT. Select a model to run (0 for Llama, 1 for Gemma): ")

which_task = input("Select a task to run (0 for creative writing, 1 for summarizing): ")
while (which_task != str(TaskChosen.CREATIVE_WRITING.value)) and (which_task != str(TaskChosen.SUMMARIZING.value)):
    which_task = input("INVALID INPUT. Select a task to run (0 for creative writing, 1 for summarizing): ")

whose_machine = input("Select whose machine is being used (0 for David, 1 for Michael): ")
while (whose_machine != str(MachineChosen.DAVID.value)) and (whose_machine != str(MachineChosen.MICHAEL.value)):
    whose_machine = input("INVALID INPUT. Select whose machine is being used (0 for David, 1 for Michael): ")

# Cast input as int so that we do not need to keep casting str(ModelChosen.___.value) for comparisons
which_model = int(which_model)
which_task = int(which_task)
whose_machine = int(whose_machine)

# Start monitoring usage stats in background
# Need to start a new thread, otherwise, program will just endlessly loop while recording usage stats
monitor_thread = threading.Thread(target=monitor_usage_to_csv, args=(which_model, which_task, whose_machine), daemon=True)
monitor_thread.start()
print("Now recording usage stats.")

# Sleep to get baseline usage stats
print("Sleeping for 10 seconds.")
time.sleep(10)

text_generation = "text-generation"
summarization = "summarization"

# Initialize pipeline
# Llama
if which_model == ModelChosen.LLAMA.value:
    # Creative Writing
    if which_task == TaskChosen.CREATIVE_WRITING.value:
        print("Initializing pipeline.")
        pipeline = generate_pipeline(model_llama)
        print(f"Model: {model_llama}")
        print(f"Task: \"{text_generation}\"")
    # Summarization
    else:
        print("Initializing pipeline.")
        pipeline = generate_pipeline(model_llama)
        print(f"Model: {model_llama}")
        print(f"Task: \"{summarization}\"")
# Gemma
else:
    # Creative Writing
    if which_task == TaskChosen.CREATIVE_WRITING.value:
        print("Initializing pipeline.")
        pipeline = generate_pipeline(model_gemma)
        print(f"Model: {model_gemma}")
        print(f"Task: \"{text_generation}\"")
    # Summarization
    else:
        print("Initializing pipeline.")
        pipeline = generate_pipeline(model_gemma)
        print(f"Model: {model_gemma}")
        print(f"Task: \"{summarization}\"")

# Start chatting
print("Starting chat.\n=============================================\n")

time_to_complete_tasks = []

# Loop for asking questions
# Creative Writing
if which_task == TaskChosen.CREATIVE_WRITING.value:
    for question in creative_writing_questions_bank:
        start_time = time.perf_counter()
        chat_with_model(pipeline, question)
        end_time = time.perf_counter()

        task_time = end_time - start_time
        time_to_complete_tasks.append(task_time)
        print(f"=============================================\nTask took {task_time:.2f} seconds to complete.\n=============================================\n")
# Summarization
else:
    for question in summarizing_questions_bank:
        start_time = time.perf_counter()
        chat_with_model(pipeline, question)
        end_time = time.perf_counter()

        task_time = end_time - start_time
        time_to_complete_tasks.append(task_time)
        print(f"=============================================\nTask took {task_time:.2f} seconds to complete.\n=============================================\n")

# Sleep to get usage stats after finishing chat
print("Sleeping for 10 seconds.")
time.sleep(10)

# Determine filename
# Model
if which_model == ModelChosen.LLAMA.value:
    filename = "llama"
else:
    filename = "gemma"

# Task
if which_task == TaskChosen.CREATIVE_WRITING.value:
    filename += "_creative_writing"
else:
    filename += "_summarization"

filename += "_latencies"

# Whose machine used
if whose_machine == MachineChosen.DAVID.value:
    filename += "_david"
else:
    filename += "_michael"

# Append .txt to filename
filename += ".txt"

with open(filename, mode="w") as file:
    # Write latencies
    file.write("Time to complete tasks (in seconds):\n")
    for data in time_to_complete_tasks:
        file.write(str(data) + '\n')

    # DEPRECATED: No longer analyzing models' responses and subjectively assessing quality of responses.
    # # Write models' responses
    # file.write("\nModel's responses:\n=============================================\n\n")
    # for response in conversation_history[1::2]:
    #     file.write(str(response) + '\n')
    #     file.write("\n=============================================\n\n")

print("\nJob's done. Goodbye.\n")