import os
import openai
import json
from tqdm import tqdm

# openai.organization = "org-cY6tsUnE8IPocUoJhZO2XbAu"  # UCLA Google
openai.organization = "org-Igmvps22Goq7QU5eddDp2SyR"  # UCLA NLP
with open('openai.key') as f:
    key = f.readline().strip()
    openai.api_key = key
# print(openai.Model.list())


MODEL="text-davinci-003"
N_SAMPLE=1
MAX_LEN=128
TEMP=0
TOP_P=1


# in_file = '/home/diwu/lm-kpgen/DeepKPG/data/scikp/kp20k/processed/test.json'
in_file = '/home/diwu/lm-kpgen/DeepKPG/data/kptimes/processed/test.json'
n_skip = 0
n_eval = 100
out_file = './jptimes_testset_zero_shot_skip{}_eval{}_maxlen{}_temp{}_topp{}.json'.format(n_skip, n_eval, MAX_LEN, TEMP, TOP_P)


# 1 - collect data
data = []
with open(in_file) as in_f:
    for _ in range(n_skip):
        _ = in_f.readline()
    
    for _ in range(n_eval):
        entry = json.loads(in_f.readline())
        data.append({'title': entry['title']['text'], 'body': entry['abstract']['text']})

# 2 - prompting
prompt_base = "Keyphrases are the phrases that summarize the most important and salient information in a document. Given a document's title and body, generate the keyphrases. You should separate the keyphrases with a comma.\n"
prompt_doc_template = "\nDocument Title: {}\nDocument Body: {}\nKeyphrases (separated by comma):"
prompt_args = {
    "model": MODEL,
    "prompt": None,
    "max_tokens": MAX_LEN,
    "temperature": TEMP,
    "top_p": TOP_P,
    "n": N_SAMPLE,
    # "stream": False,
    # "logprobs": None,
    # "stop": "\n"
}

with open(out_file, 'w') as out_f:
    for cur_doc in tqdm(data, desc='prompting'):
        prompt = prompt_base + prompt_doc_template.format(cur_doc['title'], cur_doc['body'])
        prompt_args['prompt'] = prompt
        response = openai.Completion.create(**prompt_args)
        cur_preds = {x['index']: x for x in response['choices']}
        print(json.dumps(cur_preds), file=out_f, flush=True)
