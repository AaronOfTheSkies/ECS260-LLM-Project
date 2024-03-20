from human_eval_master.human_eval.data import stream_jsonl, write_jsonl

def clean(completion):
    cleaned_string = completion.split("other than code.")[1]
    return cleaned_string

clean_samples = []

for sample in stream_jsonl("samples.jsonl"):
    clean_samples.append(dict(task_id=sample["task_id"], completion=clean(sample["completion"])))


write_jsonl("cleaned_samples.jsonl", clean_samples)