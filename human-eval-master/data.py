from human_eval.data import write_jsonl, read_problems
from transformers import AutoModelForCausalLM, AutoTokenizer
import datetime

def generate_one_completion(prompt):
#Requires installation of transformers and setuptools
    model = AutoModelForCausalLM.from_pretrained(
        "codellama/CodeLlama-7b-hf",
        local_files_only = True
    )
    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
    model_inputs = tokenizer([prompt], return_tensors="pt")
    generated_ids = model.generate(**model_inputs, max_new_tokens=200)
    results = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return results[0]


problems = read_problems()

num_samples_per_task = 200
samples = [
    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    for task_id in problems
    for _ in range(num_samples_per_task)
]
write_jsonl("samples.jsonl", samples)