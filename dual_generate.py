import argparse
import logging
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.util import load_dataset_from_path, relevant_contexts, Example, CodeBlock
from eval import compute_metric_stmt
import time
import json
import os
from datetime import datetime
from single_decoding import SingleModelDecoding
from dual_decoding import SPDecoding 
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

logger = logging.getLogger(__name__)

def get_draft_from_file(lines: list[str], current_line_index: int):
    if current_line_index == 0:
        return None, None  # 没有上一行

    find = False
    for i, line in enumerate(lines[::-1]):
        if line.strip():
            target_line = line
            # 如果上一行是长字符串，就算了，因为没法判断是长字符串的开始还是结尾
            if '"""' == target_line.strip() or "'''" == target_line.strip():
                return None, None
            former_lines = lines[:-i-1]
            find = True
            break
    
    if not find:
        return None, None
    
    for i in range(len(former_lines) - 1, -1, -1):
        # 如果上一行以回车结束，说明补全是从全新的一行开始的
        if target_line.endswith("\n"):
            # 如果有一行和上一行是一样的，那么就将它的下一行作为draft
            if former_lines[i].strip() == target_line.strip():
                j = i + 1
                # 如果下一行是空行或注释，就继续往下选
                while j < len(lines) and (not lines[j].strip() or lines[j].strip().startswith("#")):
                    j += 1
                # 如果一直都是空行，视为没找到，继续往上查找与target_line相同的行
                if j == len(lines):
                    continue
                else:
                    draft = lines[j].rstrip('\n')
                
                # 对齐空格数量
                draft_former_line_space_num = len(former_lines[i]) - len(former_lines[i].lstrip())
                draft_space_num = len(draft) - len(draft.lstrip())
                target_line_space_num = len(target_line) - len(target_line.lstrip())
                draft = " " * (target_line_space_num + draft_space_num - draft_former_line_space_num) + draft.lstrip()
                
                return draft, target_line
        # 否则补全是接着一行的前半部分进行的
        # 查找的时候要做的就是前缀匹配，然后把后缀作为draft
        else:
            if former_lines[i].lstrip().startswith(target_line.lstrip()):
                start_position = lines[i].find(target_line.strip())
                return lines[i][start_position+len(target_line.strip()):], target_line

    return None, target_line

def get_draft_from_codeblock(codeblocks, target_line):
    for cb in codeblocks:
        lines = cb.code_content.splitlines(keepends=True)
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i]
            # 跳过空行
            if not line.strip():
                continue

            # 跳过 """ 或 ''' 的单独一行
            if line.strip() in {'"""', "'''" }:
                continue

            if line.strip() == target_line.strip() and target_line.endswith("\n"):
                # 补全是从下一行开始的
                j = i + 1
                while j < len(lines) and (not lines[j].strip() or lines[j].strip().startswith("#")):
                    j += 1
                if j == len(lines):
                    continue
                draft = lines[j].rstrip('\n')

                # 对齐缩进
                target_indent = len(target_line) - len(target_line.lstrip())
                ref_indent = len(lines[i]) - len(lines[i].lstrip())
                draft_indent = len(draft) - len(draft.lstrip())

                aligned_draft = " " * (target_indent + draft_indent - ref_indent) + draft.lstrip()
                return aligned_draft

            elif line.strip().startswith(target_line.strip()) and not target_line.endswith("\n"):
                # 补全是当前行的继续内容
                start_position = line.find(target_line.strip())
                if start_position != -1:
                    return line[start_position + len(target_line.strip()):]
    
    if target_line.endswith("."):
        target_parts = target_line.split()
        target_obj = target_parts[-1]
        for cb in codeblocks:
            lines = cb.code_content.splitlines(keepends=True)
            for line in lines:
                if target_obj in line:
                    start_position = line.find(target_obj)
                    return line[start_position + len(target_obj):]
        

    return None

def format_example(args, tokenizer, example:Example):
    prefix = example.prefix
    suffix = example.suffix

    cross_file_budget = int(0.75 * args.max_input_length)
    # repo_content = cross_file_contexts(example.relevant_code, tokenizer, cross_file_budget)
    if args.stmts:
        if args.full:
            small_relevant = example.full_line_relevant_stmts
            if small_relevant==None:
                small_relevant = example.small_relevant_stmts
        else:
            small_relevant = example.small_relevant_stmts
        
    else:
        if args.full:
            small_relevant = example.full_line_relevant_codes
            if small_relevant == None:
                small_relevant = example.small_relevant_codes
        else:
            small_relevant = example.small_relevant_codes

    large_input, small_input = relevant_contexts(small_relevant[:args.relevant_code_num], example.relevant_codes[:args.relevant_code_num], \
        tokenizer, cross_file_budget, small_repo_percent=args.repo_percent) #100% repo
    
    if args.model_type == "opc":
        repo_input = small_input
    else:
        repo_input = large_input
    
    prefix_tokenized_result = tokenizer(prefix, add_special_tokens=False)
    suffix_tokenized_result = tokenizer(suffix, add_special_tokens=False)
    
    left_budget = args.max_input_length - len(repo_input["input_ids"]) - 4
    prefix_length = int(left_budget / 2)
    suffix_length = int(left_budget - prefix_length)
    if len(prefix_tokenized_result["input_ids"]) < prefix_length and len(suffix_tokenized_result["input_ids"]) < suffix_length:
        prefix_ids = prefix_tokenized_result["input_ids"]
        suffix_ids = suffix_tokenized_result["input_ids"]
    elif len(prefix_tokenized_result["input_ids"]) < prefix_length:
        prefix_ids = prefix_tokenized_result["input_ids"]
        suffix_length = int(left_budget - len(prefix_ids))
        suffix_ids = suffix_tokenized_result["input_ids"][:suffix_length]
    elif len(suffix_tokenized_result["input_ids"]) < suffix_length:
        suffix_ids = suffix_tokenized_result["input_ids"]
        prefix_length = int(left_budget - len(suffix_ids))
        prefix_ids = prefix_tokenized_result["input_ids"][-prefix_length:]
    else:
        prefix_ids = prefix_tokenized_result["input_ids"][-prefix_length:]
        suffix_ids = suffix_tokenized_result["input_ids"][:suffix_length]
    
    repo_str = tokenizer.decode(repo_input["input_ids"], skip_special_tokens=True)
    
    prefix_str = tokenizer.decode(prefix_ids, skip_special_tokens=True)
    suffix_str = tokenizer.decode(suffix_ids, skip_special_tokens=True)
    return repo_str, prefix_str, suffix_str
# --------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path",  type=str,default="/nasdata/Model/Qwen2.5-Coder-7B", help="Path to the large model.")

    parser.add_argument("--output_dir",  type=str, default='outputs', help="Path to save the generated outputs.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda or cpu).")
    parser.add_argument("--max_input_length", type=int, default=2048, help="Maximum generation length.")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--stmts",action="store_true")
    parser.add_argument("--repo_percent",type=float,required=True)
    parser.add_argument("--trigger", action="store_true", help="larger model use trigger")
    parser.add_argument("--full", action="store_true", help="use full line to retrive for small models' output")
    parser.add_argument("--one_model", action="store_true")
    parser.add_argument("--twice", action="store_true")
    parser.add_argument("--without_sp", action="store_true")
    parser.add_argument("--model_type", default=None, choices=["qwen", "starcoder", "deepseek", "opc"])
    parser.add_argument("--relevant_code_num", default=5, type=int)
    parser.add_argument("--retrieve_draft", action="store_true")
    

    args = parser.parse_args()
    print(args.stmts)
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Using device: {args.device}")
    
    if args.twice:
        assert args.one_model
    
    if args.model_type == "opc":
        # 小模型是拿不到用自己生成内容检索到的信息的
        assert args.repo_percent == 1
    
    assert args.model_type

    if args.model_type == "opc":
        logger.info(f"Loading small model from {args.model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(args.device)
    else:
        logger.info(f"Loading large model from {args.model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16).to(args.device)
    
    model.eval()
    logger.info(f"load: {args.model_name_or_path}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    ## 加载parser
    PY_LANGUAGE = Language(tspython.language())
    parser = Parser(PY_LANGUAGE)

    if args.one_model:
        all_eval_examples = {
            # "cceval": load_dataset_from_path(f"preprocessed/cceval-5.pkl"),
            "repoeval_line": load_dataset_from_path(f"preprocessed/repoeval_line-5.pkl"),
            "repoeval_api": load_dataset_from_path(f"preprocessed/repoeval_api-5.pkl"),
            "ours": load_dataset_from_path(f"preprocessed/ours-5.pkl"),
            "ours_suffix": load_dataset_from_path(f"preprocessed/ours_suffix-5.pkl"),
            "execrepoeval": load_dataset_from_path("preprocessed/execrepoeval-5.pkl"),
            # "cceval_prefix": load_dataset_from_path(f"preprocessed/cceval_only_prefix-5.pkl"),
            "repoeval_line_prefix": load_dataset_from_path(f"preprocessed/repoeval_line_only_prefix-5.pkl"),
            "repoeval_api_prefix": load_dataset_from_path(f"preprocessed/repoeval_api_only_prefix-5.pkl"),
            "ours_prefix": load_dataset_from_path(f"preprocessed/ours_only_prefix-5.pkl"),
            "ours_suffix_prefix": load_dataset_from_path(f"preprocessed/ours_suffix_only_prefix-5.pkl"),
            "execrepoeval_prefix": load_dataset_from_path("preprocessed/execrepoeval_only_prefix-5.pkl")
        }
    else:
        all_eval_examples = {
            # "cceval": load_dataset_from_path(f"preprocessed_retrieval_twice/3080ti/cceval-5.pkl"),
            "repoeval_line": load_dataset_from_path(f"preprocessed_retrieval_twice/3080ti/repoeval_line-5.pkl"),
            "repoeval_api": load_dataset_from_path(f"preprocessed_retrieval_twice/3080ti/repoeval_api-5.pkl"),
            "ours": load_dataset_from_path(f"preprocessed_retrieval_twice/3080ti/ours-5.pkl"),
            "ours_suffix": load_dataset_from_path(f"preprocessed_retrieval_twice/3080ti/ours_suffix-5.pkl"),
            "execrepoeval": load_dataset_from_path("preprocessed_retrieval_twice/3080ti/execrepoeval-5.pkl"),
            # "cceval_prefix": load_dataset_from_path(f"preprocessed_retrieval_twice/3080ti/cceval_only_prefix-5.pkl"),
            "repoeval_line_prefix": load_dataset_from_path(f"preprocessed_retrieval_twice/3080ti/repoeval_line_only_prefix-5.pkl"),
            "repoeval_api_prefix": load_dataset_from_path(f"preprocessed_retrieval_twice/3080ti/repoeval_api_only_prefix-5.pkl"),
            "ours_prefix": load_dataset_from_path(f"preprocessed_retrieval_twice/3080ti/ours_only_prefix-5.pkl"),
            "ours_suffix_prefix": load_dataset_from_path(f"preprocessed_retrieval_twice/3080ti/ours_suffix_only_prefix-5.pkl"),
            "execrepoeval_prefix": load_dataset_from_path("preprocessed_retrieval_twice/3080ti/execrepoeval_only_prefix-5.pkl")
        }
    
    for name,examples in tqdm(all_eval_examples.items(),desc="Formatting"):
        print(name)
        for example in examples:
            example.relevant_str, example.prefix_str, example.suffix_str = format_example(args, tokenizer, example)
            
    # 4. 初始化双模型解码器
    if args.one_model:
        decoder = SingleModelDecoding(
            model=model,
            model_type=args.model_type, # e.g., 'deepseek', 'qwen'
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            parser=parser,
            twice=args.twice
        )
    else:
        decoder = SPDecoding(
            model=model,
            model_type=args.model_type, # e.g., 'deepseek', 'qwen'
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            parser=parser,
            without_sp=args.without_sp
        )

    # 创建时间统计字典和结果目录
    time_stats = {}
    os.makedirs(args.output_dir, exist_ok=True)
   
    for key, examples in tqdm(all_eval_examples.items(), desc="Generating"):
        if decoder.task != key:
            decoder.set_task(key)
            print(f"Switching to {key} task")

        time_stats[key] = {}
        results = []
        logger.info("Evaluating on {} dataset".format(key))
        save_path = f"{args.output_dir}/result/{key}/prediction.jsonl"
        if not os.path.exists(save_path):
            for example in tqdm(examples):
                sample_start_time = time.time()
                if args.one_model:
                    generated_code = decoder.generate(
                        prefix=example.prefix_str,
                        suffix=example.suffix_str,
                        relevant_str=example.relevant_str,
                        ground_truth=example.middle,
                        prompt=example.prefix
                    )
                    mismatch = None
                    early_stop_time = None
                elif args.retrieve_draft:
                    draft, target_line = get_draft_from_file(example.prefix.splitlines(keepends=True), example.prefix.count("\n"))
                    if not draft and target_line:
                        draft = get_draft_from_codeblock(example.relevant_codes, target_line)
                    if not draft:
                        draft = ""
                    generated_code, mismatch, early_stop_time = decoder.generate(
                        prefix=example.prefix_str,
                        suffix=example.suffix_str,
                        relevant_str=example.relevant_str,
                        ground_truth=example.middle,
                        trigger_point_idx=example.trigger_point_idx if args.trigger else None,
                        prompt=example.prefix,
                        small_output=draft
                    )
                else:
                    generated_code, mismatch, early_stop_time = decoder.generate(
                        prefix=example.prefix_str,
                        suffix=example.suffix_str,
                        relevant_str=example.relevant_str,
                        ground_truth=example.middle,
                        trigger_point_idx=example.trigger_point_idx if args.trigger else None,
                        prompt=example.prefix,
                        small_output=example.small_pred
                    )

                sample_end_time = time.time()
                if mismatch is not None:
                    mismatch = mismatch[0].tolist()
                if args.trigger:
                    if example.trigger_point_idx != None:
                        generated_code = example.middle[:example.trigger_point_idx]+generated_code
                    
                results.append({
                    "id": example.task_id,
                    "prefix": example.prefix,
                    "suffix": example.suffix,
                    "generated": generated_code,
                    "ground_truth":example.middle,
                    "time": sample_end_time - sample_start_time,
                    "early_stop_time": sample_end_time - early_stop_time if early_stop_time else None,
                    "small_output":example.small_pred,
                    "mismatch": mismatch
                }) 

            total_time = sum(result["time"] for result in results)
            avg_time_per_sample = total_time / len(results)

            time_stats[key] = {
                "total_time": total_time,
                "samples": len(examples),
                "avg_time_per_sample":avg_time_per_sample 
            }
            
            logger.info(f"Saving results to {args.output_dir}")
            # 保存生成结果
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                for result in tqdm(results,desc='Saving'):
                    f.write(json.dumps({"task_id": result["id"], "pred": result["generated"], "time":result['time'], \
                        'small_output':result['small_output'], 'ground_truth':result['ground_truth'], \
                        'prefix':result['prefix'], 'suffix':result['suffix'], "mismatch":result["mismatch"], \
                        "early_stop_time":result["early_stop_time"]}) + "\n")
        results = compute_metric_stmt(f"{args.output_dir}/result/{key}", args)

        # 将评估结果添加到时间统计中
        time_stats[key].update(results)


    # 保存时间统计结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    time_stats_path = f"{args.output_dir}/time_stats_{timestamp}.json"
    with open(time_stats_path, "w", encoding="utf-8") as f:
        json.dump(time_stats, f, indent=4)
    
    logger.info(f"Time statistics saved to {time_stats_path}")
    logger.info("Generation finished.")

if __name__ == "__main__":
    main()