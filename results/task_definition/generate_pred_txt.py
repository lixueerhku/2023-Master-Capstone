import json
# from nltk.stem.porter import *

# clean string
def generate_word_list(str):
    str = str.lower() # to lowercase
    str = str.replace(", ", " ; ")
    word_list = str.split(" ; ")
    # stemmer = PorterStemmer()
    for i in range(len(word_list)):
        s = word_list[i]
        # word_list[i] = ' '.join([stemmer.stem(x) for x in s.lower().strip().split()]) # stem every word
        word_list[i] = ' '.join([x for x in s.lower().strip().split()])
    return ' ; '.join(word_list) # example: "a ; b ; c"

def read_pred_results(pred_file_path, n_skip=0, n_eval=1, n_prompt=6, curr_prompt=0):# every case repeat n_prompt times
    pred_results = []
    with open(pred_file_path) as in_f:
        for _ in range(n_skip):
            _ = in_f.readline()
        
        for i in range(n_eval):
            entry = json.loads(in_f.readline())
            if i % n_prompt == curr_prompt:
                pred_str = entry["0"]['text']
                pred_results.append(generate_word_list(pred_str))
    return pred_results

pred_file_path = "/Users/xueerli/Desktop/capstone/results/task_definition/e6_diversity/kp20k_task_def_diversity_zeroshot_testset_zero_shot_skip0_eval100_maxlen128_temp0_topp1.json"
total_examples = 200
num_prompt = 2

for i in range(num_prompt):
    with open('e6_kp20k_diversity_prompt_'+str(i)+'.txt', 'w') as f:
        pred_results = read_pred_results(pred_file_path, 0, total_examples, n_prompt=num_prompt, curr_prompt=i)
        for example in pred_results:
            f.write(example)
            f.write('\n')