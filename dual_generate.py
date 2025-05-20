import argparse
import logging
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.util import load_dataset_from_path, relevent_contexts, Example, CodeBlock
from eval import compute_metric_stmt
import time
import json
import os
from datetime import datetime
# gc
import gc
from single_decoding import SingleModelDecoding
from dual_decoding import SPDecoding 
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

logger = logging.getLogger(__name__)

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

    large_input, small_input = relevent_contexts(small_relevant[:args.relevant_code_num], example.relevant_codes[:args.relevant_code_num], \
        tokenizer, cross_file_budget, small_repo_percent=args.repo_percent) #100% repo
    
    if args.large_model_type is None:
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
    
    # 上下文检索的+根据小模型检索的，用于双模型解码模式
    repo_str = tokenizer.decode(repo_input["input_ids"], skip_special_tokens=False)
    
    ### 只有根据上下文检索的,用于单一模型解码模式
    # repo_str = tokenizer.decode(small_input["input_ids"], skip_special_tokens=False)  
    
    prefix_str = tokenizer.decode(prefix_ids,skip_special_tokens=True)
    suffix_str = tokenizer.decode(suffix_ids,skip_special_tokens=True)
    return repo_str, prefix_str, suffix_str
# --------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--l_model_name_or_path",  type=str,default="/data/chuyangxu/lhz/model/Qwen/Qwen2.5-Coder-7B", help="Path to the large model.")
    parser.add_argument("--s_model_name_or_path",  type=str,default="/data/chuyangxu/lhz/model/opc-sft-v1-modified", help="Path to the small model.")

    parser.add_argument("--output_dir",  type=str, default='outputs', help="Path to save the generated outputs.")
    parser.add_argument("--lang", type=str, default="python", help="Programming language.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda or cpu).")
    parser.add_argument("--max_input_length", type=int, default=2048, help="Maximum generation length.")
    parser.add_argument("--stmts",action="store_true")
    parser.add_argument("--repo_percent",type=float,required=True)
    parser.add_argument("--trigger", action="store_true", help="larger model use trigger")
    parser.add_argument("--full", action="store_true", help="use full line to retrive for small models' output")
    parser.add_argument("--one_model", action="store_true")
    parser.add_argument("--twice", action="store_true")
    parser.add_argument("--without_sp", action="store_true")
    parser.add_argument("--large_model_type", default=None, choices=["qwen", "starcoder", "deepseek", None])
    parser.add_argument("--relevant_code_num", default=5, type=int)
    

    args = parser.parse_args()
    print(args.stmts)
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Using device: {args.device}")
    
    if args.twice:
        assert args.one_model

    if args.one_model:
        if args.large_model_type:
            logger.info(f"Loading large model from {args.l_model_name_or_path}")
            l_tokenizer = AutoTokenizer.from_pretrained(args.l_model_name_or_path)
            l_model = AutoModelForCausalLM.from_pretrained(args.l_model_name_or_path,torch_dtype=torch.float16).to(args.device)
            l_model.eval()
            logger.info("Large model loaded.")
            
            if l_tokenizer.pad_token is None:
                l_tokenizer.pad_token = l_tokenizer.eos_token
            
        else:
            logger.info(f"Loading small model from {args.s_model_name_or_path}")
            s_tokenizer = AutoTokenizer.from_pretrained(args.s_model_name_or_path)
            s_model = AutoModelForCausalLM.from_pretrained(args.s_model_name_or_path).to(args.device)
            s_model.eval()
            logger.info("Small model loaded.")
            
            if s_tokenizer.pad_token is None:
                s_tokenizer.pad_token = s_tokenizer.eos_token
            
    else:
        # 1. 加载大模型和 Tokenizer
        logger.info(f"Loading large model from {args.l_model_name_or_path}")
        l_tokenizer = AutoTokenizer.from_pretrained(args.l_model_name_or_path)
        l_model = AutoModelForCausalLM.from_pretrained(args.l_model_name_or_path,torch_dtype=torch.float16).to(args.device)
        l_model.eval()
        logger.info("Large model loaded.")

        # 2. 加载小模型和 Tokenizer
        logger.info(f"Loading small model from {args.s_model_name_or_path}")
        s_tokenizer = AutoTokenizer.from_pretrained(args.s_model_name_or_path)
        s_model = AutoModelForCausalLM.from_pretrained(args.s_model_name_or_path).to(args.device)
        s_model.eval()
        logger.info("Small model loaded.")
        
        if l_tokenizer.pad_token is None:
            l_tokenizer.pad_token = l_tokenizer.eos_token
        
        if s_tokenizer.pad_token is None:
            s_tokenizer.pad_token = s_tokenizer.eos_token
    
    ## 加载parser
    PY_LANGUAGE = Language(tspython.language())
    parser = Parser(PY_LANGUAGE)

    step = "step7-more"
    all_eval_examples = {
        "ours": load_dataset_from_path(f"preprocessed_retrieval_twice/{step}/ours-5.pkl"),
        "ours_suffix": load_dataset_from_path(f"preprocessed_retrieval_twice/{step}/ours_suffix-5.pkl"),
        "cceval": load_dataset_from_path(f"preprocessed_retrieval_twice/{step}/cceval-5.pkl"),
        "repoeval_line": load_dataset_from_path(f"preprocessed_retrieval_twice/{step}/repoeval_line-5.pkl"),
        "repoeval_api": load_dataset_from_path(f"preprocessed_retrieval_twice/{step}/repoeval_api-5.pkl"),
        "ours_prefix": load_dataset_from_path(f"preprocessed_retrieval_twice/{step}/ours_only_prefix-5.pkl"),
        "ours_suffix_prefix": load_dataset_from_path(f"preprocessed_retrieval_twice/{step}/ours_suffix_only_prefix-5.pkl"),
        "cceval_prefix": load_dataset_from_path(f"preprocessed_retrieval_twice/{step}/cceval_only_prefix-5.pkl"),
        "repoeval_line_prefix": load_dataset_from_path(f"preprocessed_retrieval_twice/{step}/repoeval_line_only_prefix-5.pkl"),
        "repoeval_api_prefix": load_dataset_from_path(f"preprocessed_retrieval_twice/{step}/repoeval_api_only_prefix-5.pkl"),
    }
    
    # for testing
    # all_eval_examples = {}
    # def dict_to_code_block(data) -> CodeBlock:
    #     """
    #     Convert a dictionary to a CodeBlock object.
    #     """
    #     return CodeBlock(
    #         file_path=data["file_path"],
    #         code_content=data["code_content"]
    #     )
    
    # def dict_to_example(data) -> Example:
    #     """
    #     Convert a dictionary to an Example object.
    #     """
    #     relevant_codes = [dict_to_code_block(code_block_data) for code_block_data in (data.get("relevant_codes") or [])]
    #     full_line_relevant_stmts = [dict_to_code_block(stmt_data) for stmt_data in (data.get("full_line_relevant_stmts") or [])]
    #     full_line_relevant_codes = [dict_to_code_block(code_block_data) for code_block_data in (data.get("full_line_relevant_codes") or [])]
    #     small_relevant_stmts = [dict_to_code_block(stmt_data) for stmt_data in (data.get("small_relevant_stmts") or [])]
    #     small_relevant_codes = [dict_to_code_block(code_block_data) for code_block_data in (data.get("small_relevant_codes") or [])]
    #     small_relevant_local_stmts = [dict_to_code_block(stmt_data) for stmt_data in (data.get("small_relevant_local_stmts") or [])]
        
    #     return Example(
    #         task_id=data["task_id"],
    #         prefix=data["prefix"],
    #         suffix=data["suffix"],
    #         middle=data["middle"],
    #         relevant_codes=relevant_codes,
    #         small_pred=data["small_pred"],
    #         correct=data["correct"],
    #         full_line_relevant_stmts=full_line_relevant_stmts,
    #         full_line_relevant_codes=full_line_relevant_codes,
    #         small_relevant_stmts=small_relevant_stmts,
    #         small_relevant_codes=small_relevant_codes,
    #         small_relevant_local_stmts=small_relevant_local_stmts
    #     )

    # bn = "repoeval_api"
    # examples = []
    # with open(f"preprocessed_retrieval_twice/step7-more/{bn}-5.jsonl", 'r') as f:
    #     for line in f:
    #         example_dict = json.loads(line)
    #         examples.append(dict_to_example(example_dict))
    # all_eval_examples[f"{bn}"] = examples
    # for testing end
    
    for name,examples in tqdm(all_eval_examples.items(),desc="Formatting"):
        print(name)
        for example in examples:
            example.relevent_str, example.prefix_str, example.suffix_str = format_example(args, l_tokenizer if args.large_model_type else s_tokenizer, example)
            
    # 4. 初始化双模型解码器
    if args.one_model:
        if args.large_model_type == "qwen" or args.large_model_type == "deepseek" or args.large_model_type == "starcoder":
            decoder = SingleModelDecoding(
                model=l_model,
                model_type=args.large_model_type, # e.g., 'deepseek', 'qwen'
                tokenizer=l_tokenizer,
                device=args.device,
                lang=args.lang,
                parser=parser,
                twice=args.twice
            )
        elif args.large_model_type is None:
            decoder = SingleModelDecoding(
                model=s_model,
                model_type="opc", # e.g., 'deepseek', 'qwen'
                tokenizer=s_tokenizer,
                device=args.device,
                lang=args.lang,
                parser=parser,
                twice=args.twice
            )
        else:
            raise RuntimeError("Unsupport model type")
    else:
        decoder = SPDecoding(
            model=l_model,
            model_type=args.large_model_type, # e.g., 'deepseek', 'qwen'
            tokenizer=l_tokenizer,
            device=args.device,
            lang=args.lang,
            parser=parser,
            without_sp=args.without_sp
        )

    # 创建时间统计字典和结果目录
    time_stats = {}
    os.makedirs(args.output_dir, exist_ok=True)
    flash = 5
   
    for key, examples in tqdm(all_eval_examples.items(), desc="Generating"):
        if decoder.task != key:
            decoder.set_task(key)
            print(f"Switching to {key} task")

        time_stats[key] = {}
        i = 0
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
                        relevent_str=example.relevent_str,
                        ground_truth=example.middle,
                        prompt=example.prefix
                    )
                else:
                    generated_code = decoder.generate(
                        prefix=example.prefix_str,
                        suffix=example.suffix_str,
                        relevent_str=example.relevent_str,
                        ground_truth=example.middle,
                        trigger_point_idx=example.trigger_point_idx if args.trigger else None,
                        prompt=example.prefix,
                        small_output=example.small_pred
                    )

                sample_end_time = time.time()
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
                    "small_output":example.small_pred
                })
                i += 1
                # del generated_code
                if i % flash == 0:
                    # logger.info(f"Empty GPU memory")
                    torch.cuda.empty_cache()  
                    gc.collect()
            torch.cuda.empty_cache()  
            gc.collect()        

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
                    f.write(json.dumps({"task_id": result["id"], "pred": result["generated"],"time":result['time'],'small_output':result['small_output'], \
                        'ground_truth':result['ground_truth'],'prefix':result['prefix'],'suffix':result['suffix']}) + "\n")
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