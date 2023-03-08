import os
import openai
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
N_SAMPLE=1
MAX_LEN=128
TEMP=0
TOP_P=1


ds = 'kptimes'
if ds in ['kp20k', 'inspec', 'krapivin', 'nus', 'semeval']:
    in_file = f'/home/diwu/lm-kpgen/DeepKPG/data/scikp/{ds}/processed/test.json'
else:
    in_file = f'/home/diwu/lm-kpgen/DeepKPG/data/{ds}/processed/test.json'

n_skip = 0
n_eval = 20000
max_body_len = 500   # words
out_file = './{}_task_def_testset_zero_shot_skip{}_eval{}_maxlen{}_temp{}_topp{}.json'.format(ds, n_skip, n_eval, MAX_LEN, TEMP, TOP_P)


# 1 - collect data
data = []
with open(in_file) as in_f:
    for _ in range(n_skip):
        _ = in_f.readline()
    
    for _ in range(n_eval):
        entry = json.loads(in_f.readline())
        data.append({'title': entry['title']['text'], 'body': entry['abstract']['text']})

# 2 - prompting
prompt_bases = ["Keyphrases are the phrases that summarize the most important and salient information in a document. Given a document's title and body, generate high-level keyphrases that capture the most salient information and can be used to retrieve the document among documents from a similar domain. You should separate the keyphrases with a comma.\n",
                "Keyphrases are the phrases that summarize the most important and salient information in a document. Given a document's title and body, generate the keyphrases that contain a person's name, location, time period, topic, organization name. You should separate the keyphrases with a comma. You should separate different categories in different lines. Finally, based on the document and concepts you generate, generate high-level keyphrases that capture the most salient information and can be used to retrieve the document among documents from a similar domain.\n",
               ]

example_no_concepts_1 = ("title: e . m . t . convicted of sexual attacks on [digit] in brooklyn\n"
"Body: an emergency medical technician with the fire department was convicted on wednesday of a series of sexual assaults in brooklyn , including an attack on an [digit] - year - old girl inside an elevator . the technician , angus pascall , [digit] , was convicted of first - degree rape , among other charges , for five separate attacks on young women and girls ages [digit] to [digit] stretching to [digit] . most of the assaults occurred in [digit] and [digit] , the year he was arrested , the kings county district attorney , charles j . hynes , said in a statement . mr . pascall \u2019 s lawyer , edward friedman , said his client would appeal the verdict . in each of the attacks , mr . pascall was armed , sometimes with a gun or a knife . in one attack on a [digit] - year - old woman in [digit] , he used a machete , the district attorney said . in the assault on the [digit] - year - old , he used his emergency responder \u2019 s key to trap the victim inside an elevator . \u201c pascall then put a gun to her face and repeatedly sexually assaulted her ,\u201d according to the statement . he is scheduled to be sentenced on feb . [digit] , and he faces up to life in prison . mr . pascall had worked for the fire department for five years when he was arrested in [digit].\n"
"High-level Keyphrases: Sexual Assaults, Brooklyn, Fire Department, Angus Pascall, Charles J. Hynes, Edward Friedman, rape, angus pascall, emergency medicine, decisions and verdicts, nyc, child abuse\n"
)

example_no_concepts_2 = ("title: baylor center out of n . b . a . draft\n"
"body: baylor center isaiah austin will withdraw from the n . b . a . draft after having been found to have a rare genetic disorder . austin has marfan syndrome , a disorder that affects the connective tissue and can weaken the aorta . he announced in april that he was leaving baylor to declare for the n . b . a . draft , which is thursday.\n"
"High-level keyphrases: Isaiah Austin, Marfan Syndrome, N.B.A. Draft, Baylor Center, Connective Tissue Disorder, basketball, sports drafts and recruits\n"
)

example_no_concepts_3 = ("title: chechnya : women forced to wear head scarves , report says\n"
"body: chechnya \u2019 s strongman ramzan kadyrov is forcing women to observe an islamic dress code , while the kremlin remains silent , human rights watch said thursday . the new york - based organization said that women \u2019 s rights in the southern russian republic had deteriorated to the point that they were not allowed to enter , much less work , in government offices without head scarves , long sleeves and skirts below the knee , and that girls and young women could not attend school or university if their heads were uncovered . women are not allowed to enter movie theaters or concert halls or often even to be outdoors without head scarves , the human rights watch report said . tanya lokshina , a russia researcher at human rights watch and the report \u2019 s author , said at a news conference in moscow that the chechen women shot with paintballs last summer for failing to follow the dress code were too frightened to file formal complaints . some of the unidentified assailants were in uniform and thought to be law enforcement officials . mr . kadyrov has said that he was not behind the attacks , but that the assailants should receive awards for their deed .\n"
"High-Level Keyphrases: Women's Rights in Chechnya, Islamic Dress Code in Chechnya, Human Rights Watch Report, Attack on Women in Chechnya, Ramzan Kadyrov and Women's Rights, human rights watch,  women and girls, muslim veiling, human rights and human rights violations\n"
)

example_with_concepts_1 = ("title: e . m . t . convicted of sexual attacks on [digit] in brooklyn\n" 
"Body: an emergency medical technician with the fire department was convicted on wednesday of a series of sexual assaults in brooklyn , including an attack on an [digit] - year - old girl inside an elevator . the technician , angus pascall , [digit] , was convicted of first - degree rape , among other charges , for five separate attacks on young women and girls ages [digit] to [digit] stretching to [digit] . most of the assaults occurred in [digit] and [digit] , the year he was arrested , the kings county district attorney , charles j . hynes , said in a statement . mr . pascall \u2019 s lawyer , edward friedman , said his client would appeal the verdict . in each of the attacks , mr . pascall was armed , sometimes with a gun or a knife . in one attack on a [digit] - year - old woman in [digit] , he used a machete , the district attorney said . in the assault on the [digit] - year - old , he used his emergency responder \u2019 s key to trap the victim inside an elevator . \u201c pascall then put a gun to her face and repeatedly sexually assaulted her ,\u201d according to the statement . he is scheduled to be sentenced on feb . [digit] , and he faces up to life in prison . mr . pascall had worked for the fire department for five years when he was arrested in [digit].\n"
"Person: Angus Pascall, Charles J. Hynes, Edward Friedman\n"
"Location: Brooklyn, nyc\n"
"Time Period: Feb\n"
"Topic: Sexual Assaults\n"
"Organization Name: Fire Department\n"
"High-level Keyphrases: Sexual Assaults, Brooklyn, Fire Department, Angus Pascall, Charles J. Hynes, Edward Friedman, rape, angus pascall, emergency medicine, decisions and verdicts, nyc, child abuse\n"
)

example_with_concepts_2 = ("title: baylor center out of n . b . a . draft\n"
"body: baylor center isaiah austin will withdraw from the n . b . a . draft after having been found to have a rare genetic disorder . austin has marfan syndrome , a disorder that affects the connective tissue and can weaken the aorta . he announced in april that he was leaving baylor to declare for the n . b . a . draft , which is thursday.\n"
"Person: Isaiah Austin\n"
"Location: Baylor Center\n"
"Time Period: April\n"
"Topic: Marfan Syndrome\n"
"Organization Name: N.B.A.\n"
"High-level keyphrases: Isaiah Austin, Marfan Syndrome, N.B.A. Draft, Baylor Center, Connective Tissue Disorder, basketball, sports drafts and recruits\n"
)

example_with_concepts_3 = ("title: chechnya : women forced to wear head scarves , report says\n"
"body:chechnya \u2019 s strongman ramzan kadyrov is forcing women to observe an islamic dress code , while the kremlin remains silent , human rights watch said thursday . the new york - based organization said that women \u2019 s rights in the southern russian republic had deteriorated to the point that they were not allowed to enter , much less work , in government offices without head scarves , long sleeves and skirts below the knee , and that girls and young women could not attend school or university if their heads were uncovered . women are not allowed to enter movie theaters or concert halls or often even to be outdoors without head scarves , the human rights watch report said . tanya lokshina , a russia researcher at human rights watch and the report \u2019 s author , said at a news conference in moscow that the chechen women shot with paintballs last summer for failing to follow the dress code were too frightened to file formal complaints . some of the unidentified assailants were in uniform and thought to be law enforcement officials . mr . kadyrov has said that he was not behind the attacks , but that the assailants should receive awards for their deed .\n"
"Person: Ramzan Kadyrov, Tanya Lokshina\n"
"Location: Chechnya, Southern Russian Republic, Moscow\n"
"Time Period: thursday\n"
"Topic: Women's Rights, Islamic Dress Code \n"
"Organization Name: Human Rights Watch \n"
"High-Level Keyphrases: Women's Rights in Chechnya, Islamic Dress Code in Chechnya, Human Rights Watch Report, Attack on Women in Chechnya, Ramzan Kadyrov and Women's Rights, human rights watch,  women and girls, muslim veiling, human rights and human rights violations\n"
)

prompts_base_with_example = [prompt_bases[0]+example_no_concepts_1+example_no_concepts_2+example_no_concepts_3, prompt_bases[1]+example_with_concepts_1+example_with_concepts_2+example_with_concepts_3]
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
        # turncation
        if len(cur_doc['body'].split()) > max_body_len:
            cur_doc['body'] = ' '.join(cur_doc['body'].split()[:max_body_len])
            cur_doc['body'] += ' ...'

        # building prompt
        for prompt_base in prompts_base_with_example:
            prompt = prompt_base + prompt_doc_template.format(cur_doc['title'], cur_doc['body'])
            prompt_args['prompt'] = prompt
            # response = openai.Completion.create(**prompt_args)
            # automatically retry on rate limit failure
            response = completions_with_backoff(**prompt_args)
            cur_preds = {x['index']: x for x in response['choices']}
            print(json.dumps(cur_preds), file=out_f, flush=True)