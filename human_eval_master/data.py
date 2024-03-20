from human_eval.data import write_jsonl, read_problems
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import datetime

def generate_one_completion(prompt):
#Requires installation of transformers and setuptools
    print("womp")
    t1 = datetime.datetime.now().timestamp()
    new_prompt = "Please generate the rest of this code that fits the description in the comments. Do not say anything else other than code. \n" + prompt
    #model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    messages = [
    {"role": "user", "content": "Please complete the following code based on the comments. Do not say anything else other than code." + prompt},
    ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

    model_inputs = encodeds.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=400, pad_token_id=tokenizer.eos_token_id)
    results = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    t2 = datetime.datetime.now().timestamp() - t1
    print("Time: " + str(t2))
    f.write(str(t2))
    f.write("\n")
    print(results[0])
    return results[0]
print("here")
f = open("time.txt", "a")
print(torch.cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#"google/gemma-7b-it"
#"codellama/CodeLlama-7b-hf"
# "mistralai/Mistral-7B-Instruct-v0.2"
#"https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf"
model = AutoModelForCausalLM.from_pretrained(
    "codellama/CodeLlama-7b-Instruct-hf", device_map="auto", load_in_4bit=True, token = "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"
)
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf", token = "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")
tokenizer.pad_token = tokenizer.eos_token
print("here")
problems = read_problems()
num_samples_per_task = 3
samples = [
    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    for task_id in problems
    for i in range(num_samples_per_task)
]
write_jsonl("samples.jsonl", samples)
f.close()