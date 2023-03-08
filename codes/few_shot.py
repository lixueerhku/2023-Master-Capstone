import os
import openai
import random
import json
import backoff
from tqdm import tqdm

# openai.organization = "org-cY6tsUnE8IPocUoJhZO2XbAu"  # UCLA Google
openai.organization = "org-Igmvps22Goq7QU5eddDp2SyR"  # UCLA NLP
with open('openai.key') as f:
    key = f.readline().strip()
    openai.api_key = key
# print(openai.Model.list())

# generation parameters
MODEL="text-davinci-003"
N_EXAMPLES=5
N_SAMPLE=1
MAX_LEN=128
TEMP=0
TOP_P=1


ds = 'kptimes'
if ds in ['kp20k', 'inspec', 'krapivin', 'nus', 'semeval']:
    in_file = f'/home/diwu/lm-kpgen/DeepKPG/data/scikp/{ds}/processed/test.json'
    example_file = f'/home/diwu/lm-kpgen/DeepKPG/data/scikp/kp20k/processed/train.json'
else:
    in_file = f'/home/diwu/lm-kpgen/DeepKPG/data/{ds}/processed/test.json'
    example_file = f'/home/diwu/lm-kpgen/DeepKPG/data/{ds}/processed/train.json'

n_skip = 10096
n_eval = 4
max_body_len = 490   # words
out_file = './{}_testset_{}_shot_skip{}_eval{}_maxlen{}_temp{}_topp{}.json'.format(ds, N_EXAMPLES, n_skip, n_eval, MAX_LEN, TEMP, TOP_P)


# 1 - collect data
data = []
with open(in_file) as in_f:
    for _ in range(n_skip):
        _ = in_f.readline()
    
    for _ in tqdm(range(n_eval), desc='Reading cases'):
        entry = json.loads(in_f.readline())
        # truncation
        if len(entry['abstract']['text'].split()) > max_body_len:
            entry['abstract']['text'] = ' '.join(entry['abstract']['text'].split()[:max_body_len])
            entry['abstract']['text'] += ' ...'

        data.append({'title': entry['title']['text'], 
                     'body': entry['abstract']['text']})

examples = []
with open(example_file) as in_f:
    for line in tqdm(in_f.readlines(), desc='Reading examples'):
        entry = json.loads(line)
        # truncation
        if len(entry['abstract']['text'].split()) > max_body_len:
            entry['abstract']['text'] = ' '.join(entry['abstract']['text'].split()[:max_body_len])
            entry['abstract']['text'] += ' ...'

        examples.append({'title': entry['title']['text'], 
                         'body': entry['abstract']['text'],
                         'keyphrases': ', '.join(entry['present_kps']['text'] + 
                                                 entry['absent_kps']['text'])})

# 2 - prompting
prompt_base = "Keyphrases are the phrases that summarize the most important and salient information in a document. Given a document's title and body, generate the keyphrases. You should separate the keyphrases with a comma.\n"
example_template = "\nDocument Title: {}\nDocument Body: {}\nKeyphrases (separated by comma): {}\n"
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

# @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, 
                                     openai.error.ServiceUnavailableError))
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

with open(out_file, 'w') as out_f:
    for cur_doc in tqdm(data, desc='prompting'):
        # building prompt
        prompt = prompt_base
        cur_examples = random.sample(examples, N_EXAMPLES)
        for e in cur_examples:
            prompt += example_template.format(e['title'], e['body'], e['keyphrases'])
        prompt += prompt_doc_template.format(cur_doc['title'], cur_doc['body'])
        prompt_args['prompt'] = prompt

        # request
        # response = openai.Completion.create(**prompt_args)
        # automatically retry on rate limit failure
        response = completions_with_backoff(**prompt_args)
        cur_preds = {x['index']: x for x in response['choices']}
        print(json.dumps(cur_preds), file=out_f, flush=True)