import os,sys
import yaml
import json
import numpy as np
import transformers
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_trials", type=int, default=3, help="The number of trials")
parser.add_argument("--num_steps", type=int, default=50, help="The number of steps")
parser.add_argument("--model", type=str, default="gpt-3.5-turbo-instruct", choices=["gpt-3.5-turbo-instruct", "gpt-4-0613", "meta-llama/Llama-2-13b-chat-hf"], help="The model name")
parser.add_argument("--output", type=str, default="output", help="The output folder")
parser.add_argument("--emb_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", choices=["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L12-v2"], help="The model name")
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

with open('./configs/base_config.yaml') as reader:
    config = yaml.safe_load(reader)

ret_key_examples = open('prompts/retrieval_prompt.txt').readlines()
ret_key_examples = ''.join(ret_key_examples)

if 'Llama-2' in args.model:
    # llama2
    from transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer

    model_name = args.model
    model = AutoModelForCausalLM.from_pretrained(
        model_name, load_in_4bit=True, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
elif 'gpt' in args.model:
    #openai
    import openai
    from openai import OpenAI
    os.environ["OPENAI_API_KEY"] = open('OpenAI_api_key.txt').readline()
    openai.api_key = os.environ["OPENAI_API_KEY"]
    client = OpenAI()
    
def llm(prompt, stop=["\n"]):
    
    if 'Llama-2' in args.model:
        sequences = pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=200,
            return_full_text=False,
        )
        text = sequences[0]['generated_text']
    elif 'gpt-3.5-turbo-instruct' == args.model:
        response = client.completions.create(
            model='gpt-3.5-turbo-instruct',
            prompt=prompt,
            temperature=0.0,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop
        )
        text = response.choices[0].text
    elif 'gpt-4-0613' == args.model:
        completion = client.chat.completions.create(
            model="gpt-4-0613",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for household task."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop
        )
        text = completion.choices[0].message.content

    if stop:
        text = text.split('\n')[0]
    if len(text) > 0 and text[0]=='>':
        text = text[1:]
    if len(text) > 0 and text[-1]=='.':
        text = text[:-1]
    return text.strip()

# text embedding model
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
model_embedding = SentenceTransformer(args.emb_model)

import alfworld
import alfworld.agents.environment

def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob

prefixes = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}
with open('./prompts/alfworld_3prompts.json', 'r') as f:
    d_react = json.load(f)

def generate_embeddings(memory):
    print('num_retrieval',len(memory))
    embeddings = {}
    for key in ['Task', 'Category', 'Plan', 'Actions']:
        if key=='Actions':
            retrieve_info = [m[key].copy() for m in memory]
            for i in range(len(retrieve_info)):
                for j in range(len(retrieve_info[i])):
                    retrieve_info[i][j] = retrieve_info[i][j].strip()
            embeddings[key] = [model_embedding.encode(r) for r in retrieve_info]
            continue
        retrieve_info = [m[key] for m in memory]
        # extract embeddings
        embeddings[key] = model_embedding.encode(retrieve_info)
    return embeddings

def generate_examples(query, memory, embeddings, k=3, for_plan=False, act_len=0, mode='act', key=''):
    # similarity on task, category, and plan
    cos_scores_sum = []
    for key_tmp in ['Task', 'Category', 'Plan']:
        if query[key_tmp]=='': continue
        with torch.no_grad():
            query_embeddings = model_embedding.encode([query[key_tmp]])
        cos_scores = cos_sim(query_embeddings, embeddings[key_tmp])[0]
        cos_scores_sum.append(cos_scores.tolist())
    cos_scores_sum = torch.sum(torch.tensor(cos_scores_sum), 0)
    # retrieve examples for overall plan
    if for_plan:
        _, hits = torch.topk(cos_scores_sum, k=k)
        ret_examples = [ 'Your task is to: ' + memory[h]['Task'] + '\n> ' + memory[h]['Plan'] + '\n' for h in hits]
        return ret_examples
    # similarity on action or observation
    ret_scores=[]
    ret_index=[]
    with torch.no_grad():
        query_embeddings = model_embedding.encode([key])
    for emb in embeddings['Actions']:
        if key=='':
            ret_scores.append(0)
            ret_index.append(0)
            continue
        elif mode=='act':
            # pick up action embeddings
            log_embeddings = emb[::2]
        elif mode=='obs':
            # pick up observation embeddings
            log_embeddings = emb[1::2]
        cos_scores = cos_sim(query_embeddings, log_embeddings).numpy()
        ret_scores.append(np.max(cos_scores))
        ret_index.append(np.argmax(cos_scores)*2)
    ret_scores = torch.FloatTensor(ret_scores)
    # retrieve examples for action or action plan
    _, hits = torch.topk(ret_scores+cos_scores_sum, k=k)
    ret_examples = []
    for h in hits:
        part = (max(0,ret_index[h]-act_len),min(len(memory[h]['Actions']),ret_index[h]+act_len))
        ret_examples.append('Task: ' + memory[h]['Task'] + '\nPlan: ' + memory[h]['Plan'] + '\n' + '\n'.join(memory[h]['Actions'][part[0]:part[1]]) + '\n')
    return ret_examples

def alfworld_run_react(prompt, ob, category, to_print=True):
    target_task = ob.split('\n')[1].split(': ')[1]
    init_prompt = prompt + ob + '\n>'
    prompt = ''
    actions = []
    if to_print:
        print(ob)
        sys.stdout.flush()
    for i in range(1, args.num_steps):
        action = llm(init_prompt + prompt[-4096:])
        observation, reward, done, info = env.step([action])
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        if action.startswith('think:'):
            observation = 'OK.'
        if to_print:
            print(f'Act {i}: {action}\nObs {i}: {observation}')
            sys.stdout.flush()
        prompt += f' {action}\n{observation}\n>'
        actions.append('> '+action)
        actions.append(observation)
        if done:
            # extract overall plan
            for act in actions:
                if 'think: ' in act:
                    plan = act.split('think: ')[1].strip()
                    break
            actions = actions[2:]
            # remove invalid actions and observations
            inv_act_idx = np.where(np.array(actions)=='Nothing happens.')[0]
            inv_act_idx = np.append( inv_act_idx, inv_act_idx-1)
            actions = [actions[i] for i in range(len(actions)) if i not in inv_act_idx]
            # remove locations in reasoning
            actions = [ a.split('.')[0] if 'likely to appear' in a else a for a in actions]
            data = {
                'Task': target_task,
                'Category': category,
                'Plan': plan,
                'Actions': actions,
            }
            return reward, data
    return 0, ''

def alfworld_run_rap(prompt, ob, category, memory, embeddings, to_print=True):
    # init
    if to_print:
        print(ob)
        sys.stdout.flush()
    ob_prompt = 'Here is the task information.\n' + ob.split('\n')[0] + '\n'
    target_task = ob.split('\n')[1].split(': ')[1]

    # planning
    examples = generate_examples({'Task': target_task, 'Category': category, 'Plan': ''}, memory, embeddings, k=3, for_plan=True)
    examples = 'Here are examples.\n' + ''.join(examples)
    target_prompt = '\n' + 'Here is the task. Please make a plan from the examples.\n' + 'Your task is to: ' + target_task + '\n' + '> think: To solve the task,'
    full_prompt = examples + target_prompt
    plan = llm(full_prompt[-4096:])
    plan = 'To solve the task, '+plan.split('.')[0]+'.'
    print('Plan: ' + plan)
    
    target_prompt = 'Here is the task. Please make an action from the examples.\nTask : ' + target_task + '\nPlan : ' + plan + '\n'
    # data
    data = {
        'Task': target_task,
        'Category': category,
        'Plan': plan,
        'Actions': '',
    }
    actions = []
    # extract examples
    search_object = ''
    reasoning = ''
    ret_examples = generate_examples(data, memory, embeddings, k=4, act_len=20)
    for i in range(1, args.num_steps):
        # generate action with retrieval
        examples = 'Here are examples.\n' + ''.join(ret_examples)
        full_prompt = ob_prompt + examples + target_prompt + '\n'.join(data['Actions']) + '\n>'
        action = llm(full_prompt[-4096:])
            
        # input action into alfworld
        observation, reward, done, info = env.step([action])
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        if action.startswith('think:'):
            observation = 'OK.'
        
        if to_print:
            print(f'Act {i}: {action}\nObs {i}: {observation}')
            sys.stdout.flush()
        # generate retrieval key
        if 'think:' in action:
            full_prompt = 'Here are examples.\n' + ret_key_examples + '\nHere is the task. Please make a plan from the examples.\n' + action + '\n>'
            retrieve_key = llm(full_prompt[-4096:])
            print('Retrieval key',retrieve_key)
            if 'search:' in retrieve_key:
                search_object = retrieve_key.split('search:')[1].strip()
                ret_examples = generate_examples(data, memory, embeddings, k=8, act_len=10, mode='obs', key = search_object)
            elif 'action:' in retrieve_key:
                reasoning = retrieve_key.split('action:')[1].strip()
                ret_examples = generate_examples(data, memory, embeddings, k=4, act_len=20, mode='act', key = reasoning)

        # add action and observation
        actions.append('> '+action)
        actions.append(observation)
        data['Actions'] = actions[-10:]
        # finish
        if done:
            # remove invalid actions and observations
            inv_act_idx = np.where(np.array(actions)=='Nothing happens.')[0]
            inv_act_idx = np.append( inv_act_idx, inv_act_idx-1)
            actions = [actions[i] for i in range(len(actions)) if i not in inv_act_idx]
            # update actions for memory
            data['Actions'] = actions
            return reward, data
    return 0, ''

rs_trials = []
for trial in range(args.num_trials):
    print('### trial '+str(trial+1)+' ###')
    split = "eval_out_of_distribution"
    env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=1)
    
    cnts = [0] * 6
    rs = [0] * 6
    rs_games = []
    if trial != 0:
        memory = current_memory[:]
        embeddings = generate_embeddings(memory)
    current_memory = []
    for _ in range(134):
        ob, info = env.reset()
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
        print(name)
        for i, (k, v) in enumerate(prefixes.items()):
            if name.startswith(k):
                if trial == 0:
                    prompt = 'Interact with a household to solve a task. Here are two examples.\n' + d_react[f'react_{v}_1'] + d_react[f'react_{v}_0'] + '\nHere is the task.\n'
                    r, mem_data = alfworld_run_react(prompt, ob, v)
                    if not mem_data=='':
                        current_memory.append(mem_data)
                else:
                    r, mem_data = alfworld_run_rap('', ob, v, memory, embeddings)
                    if not mem_data=='':
                        current_memory.append(mem_data)
                rs[i] += r
                cnts[i] += 1
                break
        rs_games.append(r)
        print(_+1, 'r', r, 'rs', rs, 'cnts', cnts, 'sum(rs)/sum(cnts)', sum(rs) / sum(cnts))
        print('------------\n')
    
    rs_trials.append(rs_games)
    rs_trials_max = np.max(np.array(rs_trials), axis=0)
    print('trial:', trial+1, 'success rate:', np.sum(rs_trials_max) / rs_trials_max.shape[0])
    with open(args.output+'/memory_'+str(trial+1)+'.json', 'w') as f:
        json.dump(current_memory, f, indent=4)
np.savetxt(args.output+'/result.txt', np.array(rs_trials).T, fmt='%d')
