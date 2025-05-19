# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import editdistance
from collections import defaultdict


def load_jsonl(fname):
    with open(fname, 'r', encoding='utf8') as f:
        lines = []
        for line in f:
            lines.append(json.loads(line))
        return lines

def compute_EM(target, predictions, passk):
    if len(target) == 1:
        for p in predictions[:passk]:
            if len(p) == 0:
                if target == "\n":
                    return True
                else:
                    return False
            else:
                if p[0] == target:
                    return True
                else:
                    return False
        return False
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    EM_scores = []
    for prediction in predictions[:passk]:
        prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()][:len(target_lines)]
        if len(target_lines) != len(prediction_lines):
            EM_scores.append(0)
            continue
        if target_lines == prediction_lines:
            EM_scores.append(1)
            continue
        EM_scores.append(0)
    return any(EM_scores)

def compute_ES(target, predictions, passk):
    if len(target) == 1:
        for p in predictions[:passk]:
            if len(p) == 0:
                if target == "\n":
                    return 1
                else:
                    return 0
            else:
                if p[0] == target:
                    return 1
                else:
                    return 0
        return 0
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    target_str = '\n'.join(target_lines)
    ES_scores = []
    for prediction in predictions[:passk]:
        prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()][:len(target_lines)]
        prediction_str = '\n'.join(prediction_lines)
        ES_scores.append(
            1 - (editdistance.eval(target_str, prediction_str) / max(len(target_str), len(prediction_str)))
        )
    return max(ES_scores)

def compute_score_by_repo_with_metadata(lines, stype, passk=1):
    scores = defaultdict(list)
    for line in lines:
        repo = line['task_id'].split('/')[0]
        # if repo not in repos:
        #     continue
        samples = [line['pred']]
        if stype == 'EM':
            score = compute_EM(line['target'], samples, passk)
        elif stype == 'ES':
            score = compute_ES(line['target'], samples, passk)
        scores[repo].append(score)
    avg_scores = {repo: round(sum(scores[repo]) / len(scores[repo]), 4) for repo in scores}
    repo_count = {repo: len(scores[repo]) for repo in scores}

    return sum(avg_scores.values()) / len(avg_scores.values()), sum(repo_count.values())

def eval_repoeval(file_path):
    em_ratio, total = compute_score_by_repo_with_metadata(load_jsonl(file_path), 'EM', passk=1)
    edit_sim, total = compute_score_by_repo_with_metadata(load_jsonl(file_path), 'ES', passk=1)
    return {
        "em": round(em_ratio * 100, 4),
        "es": round(edit_sim * 100, 4),
        "id_em": '-',
        "id_f1": '-',
        "total": total,
    }

if __name__ == '__main__':
    file_path = '/data2/hanzhenlu/JointRepo/test.jsonl'
    print(eval_repoeval(file_path))