import os
import pandas as pd
import json
#define path
arch = 'framework'
dataset_tags = 'GQA'
tags = 'transh4'
time = 'latest'
judge = 'loose' #or 'strict'
def get_latest_folder(path: str) -> str:
    return max([os.path.join(path, d) for d in os.listdir(path)], key=os.path.getctime)

if time == 'latest':
    latest_folder = get_latest_folder(f'/.do/results/{dataset_tags}')
    latest_log_folder = get_latest_folder('/.do/logs')
    json_path = f'{latest_folder}/json'
    log_path = f'{latest_folder}/logs/process_{dataset_tags}.log'
else:
    json_path = f'/.do/results/{dataset_tags}/{time}/json'
    log_path= f'/.do/results/{dataset_tags}/{time}/logs/process_{dataset_tags}.log'

print(latest_folder)
json_dir = json_path
# log_file= '/.do/logs/2024-01-06-gqa-viper.log'
if 'GQA' in dataset_tags:
    gt_file = '/.do/data/sample_data/code_vqa/cache/gqa/annotations/val_balanced_questions_sample2000.json'
else:
    raise NotImplementedError
csv_file_name = f'{dataset_tags}-{arch}-{time}-{judge}-{tags}'

#init variable
error_idx = []
all_samples = {}
fail_idx = []
success_idx = []

# get result
with open(log_path,'r') as f:
    log_content = f.readlines()
for line in log_content:
    if line.startswith('2024') or line.startswith('2023'):
        # sample_id = line.split(':')[3]
        # print(sample_id)
        # sample_id = sample_id.strip().split(' ')[1]
        # info = line.split(':')[4]
        sample_id = line.split(':')[2]
        sample_id = sample_id.strip().split(' ')[-1]
        info = line.strip().split(':')[3]
        try:
            state,res = info.split(',',maxsplit=1)
        except Exception as e:
            print(e)
        if res.startswith('Error'):
            error_idx.append(sample_id)
        all_samples[sample_id]=(state,res)
        if 'fail' in state:
            fail_idx.append(sample_id)
        elif 'success' in state:
            success_idx.append(sample_id)

print(f'Error sample nums:{len(error_idx)}')
print(f'Fail sample nums:{len(fail_idx)}')
#get pred res
pred_ans = []
all_id = []
for i in range(2000):
    sample_id = str(i)
    all_id.append(sample_id)
    pred = all_samples.get(sample_id)
    if pred is None:
        pred = 'Error'
        all_samples[sample_id] = pred
    else:
        pred = pred[1]
    pred_ans.append(pred)

#get gt res
gt_ans = []
with open(gt_file,'r') as f:
    content = json.load(f)
    for i,item in enumerate(content):
        a = item['answer']
        gt_ans.append(a)

all_samples_judge_res = {}
#get judge and concat result and deatil information
def judge_function(pred_ans:list,gt_ans:list,sample_id:list):
    from lavis.common.vqa_tools.vqa_eval import VQAEval
    vqa_tool = VQAEval()
    acc = []
    for pred,gt,id in zip(pred_ans,gt_ans,sample_id):
        pred = str(pred).lower()
        pred = pred.replace('true','yes').replace('false','no')
        gt = str(gt).lower()
        pred = vqa_tool.processPunctuation(pred)
        pred = vqa_tool.processDigitArticle(pred)
        vqa_acc = 0
        #初步判断
        if pred == gt:
            vqa_acc = 1
        #再次诊断
        elif gt=='yes' or gt=='no':
            if gt == 'yes' and gt in pred and 'no' not in pred:
                vqa_acc = 1
            if gt == 'no' and gt in pred and 'yes' not in pred:
                vqa_acc = 1
        elif gt in pred:
            # print(f'gt:{gt}')
            # print(f'pred:{pred}')
            if judge=='loose':
                vqa_acc = 1
            else:
                vqa_acc = 0

        all_samples_judge_res[id] = vqa_acc
        acc.append(vqa_acc)

    res = sum(acc)/len(acc)
    print(res)
    return res,acc
res,acc = judge_function(pred_ans,gt_ans,all_id)
print(f'Total acc:{res}')

# get exist detail information
dir = os.listdir(json_dir)
result = pd.DataFrame()


for sample_id in all_id:
    fname = f'{sample_id}.json'
    if fname in dir:
        with open(os.path.join(json_dir,fname)) as f:
            content = json.load(f)
            image_path = content['image_path']
            query = content['query']
            label = content['label']
            answer = content['answer']
            pred_ans.append(answer)
            code_generate_time = content['code_generate_time']
            replan_remain_time = content['replan_remain_times']
            simple_query_check_count = content['simple_query_check_count']
            valid_simple_query_check_count = content['valid_query_check_count']
            refine_time = content['refine_remian_time']
            judge_res = all_samples_judge_res[sample_id]
            outputs=dict(
                sample_id=sample_id,
                query =query,
                image_path =image_path,
                label = label,
                answer = all_samples[sample_id][1],
                code_generate_time = code_generate_time,
                replan_remain_time = replan_remain_time,
                refine_reamain_time = refine_time,
                judge_res = judge_res
            )
            result = result._append(outputs,ignore_index=True)
    else:
        outputs=dict(
                sample_id=sample_id,
                answer ='Error',
                judge_res = 0
            )
        result = result._append(outputs,ignore_index=True)

result.to_csv(f'{csv_file_name}.csv')

