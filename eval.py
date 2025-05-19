import json
import torch.multiprocessing as mp
from fuzzywuzzy import fuzz
from tqdm import tqdm
from functools import partial
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
from datasets import load_dataset
from utils.eval_repoeval import eval_repoeval
from utils.eval_util import process_examples

def cal_edit_sim(references, hypotheses):
    total = len(references)
    edit_sim = 0.0
    for pred, gt in zip(hypotheses, references):
        pred = pred.strip()
        gt = gt.strip()
        edit_sim += fuzz.ratio(pred, gt)
    return edit_sim / total

def compute_metric_stmt(output_dir, args, language="python"):
    task_ids = {}
    with open(f"{output_dir}/prediction.jsonl", "r") as f_pred:
        samples = []
        for l in f_pred.readlines():
            samples.append(json.loads(l))
            task_ids[json.loads(l)["task_id"]] = 1

    ts_lang = "c_sharp" if language == "csharp" else language

    truncated_samples = []
    em_labels = []

    if ts_lang == "python":
        PY_LANGUAGE = Language(tspython.language())
        parser = Parser(PY_LANGUAGE)
    else:
        raise Exception("unsupport language in compute_metric_stmt")
    worker = partial(process_examples, ts_lang,parser)

    for sample in tqdm(samples):
        # 直接调用 worker 函数处理数据
        trunc_s, em_label = worker(sample)
        em_labels.append(em_label)
        truncated_samples.append(trunc_s)

    exact_match = 0
    with open(f"{output_dir}/prediction_truncated.jsonl", 'w', encoding="utf-8") as pt, \
            open(f"{output_dir}/exact_match_idx.jsonl", 'w') as em:
        for trunc_s, em_label in zip(truncated_samples, em_labels):
            pt.write(json.dumps(trunc_s) + "\n")
            if em_label == 1:
                em.write(f'{trunc_s["task_id"]}\n')
                exact_match += 1

    ### Score calculation

    edit_similarities = []
    detailed_results = []

    for idx, trunc_s in enumerate(truncated_samples):
        es = cal_edit_sim([trunc_s["target"]], [trunc_s["pred"]])
        edit_similarities.append(es)

        detailed_results.append({
            "task_id": trunc_s["task_id"],
            "em": em_labels[idx],
            "es": es,
            "time":trunc_s["time"],
        })

    em_ratio = round(exact_match / len(samples) * 100, 4)
    edit_sim = round(sum(edit_similarities) / len(edit_similarities), 4)
    time = round(sum([trunc_s["time"] for trunc_s in truncated_samples]) / len(truncated_samples), 4)

    with open(f"{output_dir}/detailed_results.json", 'w') as f:
        for dr in detailed_results:
            f.write(json.dumps(dr) + "\n")

    eval_results = eval_repoeval(f"{output_dir}/prediction_truncated.jsonl")
    em_ratio = f"{em_ratio}({eval_results['em']})"
    edit_sim = f"{edit_sim}({eval_results['es']})"
    
    
    # write the results to a file
    with open(f"{output_dir}/results.json", 'w') as f:
        res = {
            "em": em_ratio,
            "es": edit_sim,
            "time": time,
            "total": len(truncated_samples)
        }
        f.write(json.dumps(res, indent=2))
    return {
        "em": em_ratio,
        "es": edit_sim,
        "total": len(truncated_samples)
    }