from human_eval_master.human_eval.data import stream_jsonl

categories = open("categorized.txt")
math = []
string_manip = []
sorting = []
math_pass = 0
string_manip_pass = 0
sorting_pass = 0
total_math = 0
total_string_manip = 0
total_sorting = 0

num_task = 0
for task in categories:
    if task.find("Mathematics - Arithmetic") != -1:
        math.append("HumanEval/" + str(num_task))
    if task.find("String Manipulation") != -1:
        string_manip.append("HumanEval/" + str(num_task))
    if task.find("Sorting") != -1:
        sorting.append("HumanEval/" + str(num_task))
    num_task += 1

math_unique = math.copy()
string_manip_unique = string_manip.copy()
sorting_unique = sorting.copy()
math_pass_unique = 0
string_manip_pass_unique = 0
sorting_pass_unique = 0

for line in stream_jsonl("cleaned_samples.jsonl_results.jsonl"):
    if line["task_id"] in math:
        if line["passed"]:
            math_pass += 1
        total_math += 1
    if line["task_id"] in string_manip:
        if line["passed"]:
            string_manip_pass += 1
        total_string_manip += 1
    if line["task_id"] in sorting:
        if line["passed"]:
            sorting_pass += 1
        total_sorting += 1
    if line["task_id"] in math_unique and line["passed"]:
        math_pass_unique += 1
        math_unique.remove(line["task_id"])
    if line["task_id"] in string_manip_unique and line["passed"]:
        string_manip_pass_unique += 1
        string_manip_unique.remove(line["task_id"])
    if line["task_id"] in sorting_unique and line["passed"]:
        sorting_pass_unique += 1
        sorting_unique.remove(line["task_id"])

print("Math (All samples): " + str(math_pass) + " out of " + str(total_math))
print("String Manipulation (All samples): " + str(string_manip_pass) + " out of " + str(total_string_manip))
print("Sorting (All samples): " + str(sorting_pass) + " out of " + str(total_sorting))
print("Math (at least one): " + str(math_pass_unique) + " out of " + str(len(math)))
print("String Manipulation (at least one): " + str(string_manip_pass_unique) + " out of " + str(len(string_manip)))
print("Sorting (at least one): " + str(sorting_pass_unique) + " out of " + str(len(sorting)))
