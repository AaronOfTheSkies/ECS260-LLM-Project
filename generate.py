from transformers import AutoModelForCausalLM, AutoTokenizer
import datetime

t1 = datetime.datetime.now().timestamp()
#Requires installation of transformers and setuptools
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token

model_inputs = tokenizer(['#Write a NumPy program to repeat elements of an array.', "#Write a Python program to count the occurrences of each word in a given sentence.", "#Write a Python program to print out nth number of the Fibonacci sequence given n as input."], return_tensors="pt", padding=True)
generated_ids = model.generate(**model_inputs, max_new_tokens=200)
results = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

for result in results:
    print(result)
    print("\n")
t2 = datetime.datetime.now().timestamp() - t1
print("Time: " + str(t2))
