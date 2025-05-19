import jsonlines
import json

def read_jsonl(file_path):
    data = []
    with jsonlines.open(file_path, 'r') as reader:
        for item in reader:
            data.append(item)
    return data
def save_jsonl(data, file_path):
    with jsonlines.open(file_path, 'w') as writer:
        writer.write_all(data)
        
def get_pass_idx(dataset):
    pass_idx = []
    for i in range(len(dataset)):
        if dataset[i]['em'] == 1:
            pass_idx.append(dataset[i]['task_id'])
    return pass_idx

def cal_two_stage_metric(em_list,all_list,small_data_pass_idx,small_data,large_data):
    em = len(em_list) / len(all_list)
    small_es = []
    small_time = []
    for i in range(len(small_data_pass_idx)):
        for item in small_data:
            if item['task_id'] == small_data_pass_idx[i]:
                small_es.append(item['es'])
                # small_time.append(item['time'])
        # small_es.append(small_data[small_data_pass_idx[i]]['es'])
    left_task = list(set(all_list) - set(small_data_pass_idx))
    left_es = []
    left_time = []
    for i in range(len(left_task)):
        for item in large_data:
            if item['task_id'] == left_task[i]:
                left_es.append(item['es'])
                # left_time.append(item['time'])
        # left_es.append(large_data[left_task[i]]['es'])
    all_es = small_es + left_es
    all_time = small_time + left_time
    # 计算pingjun
    es = sum(all_es) / len(all_es)
    # time = sum(all_time) / len(all_time)
    time = 0
    return em,es,time
    
    
    

def get_combine_res(small_dir,large_dir,dataset_name):
    res = {}
    
    # for dataset_name in dataset_names:
    small_file = small_dir + f"/{dataset_name}/detailed_results.json"
    large_file = large_dir + f"/{dataset_name}/detailed_results.json"
    small_data = read_jsonl(small_file)
    large_data = read_jsonl(large_file)
    all_index = [i['task_id'] for i in small_data]
    # 并集
    small_data_pass_idx = get_pass_idx(small_data)
    large_data_pass_idx = get_pass_idx(large_data)
    union_idx = list(set(small_data_pass_idx) | set(large_data_pass_idx))
    em,es,time = cal_two_stage_metric(union_idx,all_index,small_data_pass_idx,small_data,large_data)
    print('union:',dataset_name,em,es,time)
    res[dataset_name] = {'em':em,'es':es,'time':time}
    
    with open('combine_res.json','w') as f:
        json.dump(res,f,indent=4)
        
        
        
            
            
if __name__ == '__main__':
    small_dir = '/data/lishuifan/SmallAndLarge/output_small/result'
    dataset_names = ["cceval_python","repoeval_line","ours","ours_suffix"]

    large_dir = "/data/lishuifan/SmallAndLarge/output_relevent_old/result"
    for dataset_name in dataset_names:
        get_combine_res(small_dir,large_dir,dataset_name)