from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness
from transformers import AutoModelForCausalLM, AutoTokenizer
import datetime

def main():
    if __name__ == '__main__':
        evaluate_functional_correctness("samples.jsonl")
f = open("time.txt", "a")
model = AutoModelForCausalLM.from_pretrained(
        "codellama/CodeLlama-7b-hf",
        local_files_only = True
    )
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")

def generate_one_completion(prompt):
#Requires installation of transformers and setuptools
    t1 = datetime.datetime.now().timestamp()
    model_inputs = tokenizer(prompt, return_tensors="pt")
    generated_ids = model.generate(**model_inputs, max_new_tokens=500, pad_token_id=tokenizer.eos_token_id)
    results = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    t2 = datetime.datetime.now().timestamp() - t1
    print("Time: " + str(t2))
    f.write(str(t2))
    f.write("\n")
    return results[0]


problems = read_problems()

num_samples_per_task = 200
samples = [
    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    for task_id in problems
]
write_jsonl("samples.jsonl", samples)
f.close()