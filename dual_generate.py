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
from decoding import SingleModelDecoding
from single_decoding import SPDecoding 
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

logger = logging.getLogger(__name__)

def format_example(args, tokenizer,special_token_ids,example: Example):
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

    large_input, small_input = relevent_contexts(small_relevant, example.relevant_codes, tokenizer, cross_file_budget, small_repo_percent=args.repo_percent) #100% repo
    
    prefix_tokenized_result = tokenizer(prefix, add_special_tokens=False)
    suffix_tokenized_result = tokenizer(suffix, add_special_tokens=False)
    
    left_budget = args.max_input_length - len(small_input["input_ids"]) - 4
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
    
    input_id = [special_token_ids["suffix_id"]] + suffix_ids + [special_token_ids["prefix_id"]] + small_input["input_ids"] + prefix_ids + [special_token_ids["middle_id"]]
    # 上下文检索的+根据小模型检索的，用于双模型解码模式
    repo_str = tokenizer.decode(large_input["input_ids"], skip_special_tokens=False)
    
    ### 只有根据上下文检索的,用于单一模型解码模式
    # repo_str = tokenizer.decode(small_input["input_ids"], skip_special_tokens=False)  
    
    prefix_str = tokenizer.decode(prefix_ids,skip_special_tokens=True)
    suffix_str = tokenizer.decode(suffix_ids,skip_special_tokens=True)
    return torch.tensor(input_id, dtype=torch.long), repo_str, prefix_str, suffix_str
# --------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--l_model_name_or_path",  type=str,default="/data/chuyangxu/lhz/model/Qwen/Qwen2.5-Coder-7B", help="Path to the large model.")
    parser.add_argument("--s_model_name_or_path",  type=str,default="/data/chuyangxu/lhz/model/opc-sft-v1-modified", help="Path to the small model.")

    parser.add_argument("--output_dir",  type=str,default='output', help="Path to save the generated outputs.")
    parser.add_argument("--lang", type=str, default="python", help="Programming language.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda or cpu).")
    parser.add_argument("--max_input_length", type=int, default=2048, help="Maximum generation length.")
    parser.add_argument("--stmts", type=str, default='false')
    parser.add_argument("--repo_percent",type=float,required=True)
    parser.add_argument("--trigger", type=str, default='false', help="larger model use trigger")
    parser.add_argument("--full", type=str, default='false', help="use full line to retrive for small models' output")
    parser.add_argument("--one_model", action="store_true")
    parser.add_argument("--twice", action="store_true")
    parser.add_argument("--without_sp", action="store_true")
    parser.add_argument("--large_model_type", default=None, choices=["qwen", "starcoder", "deepseek", None])
    

    args = parser.parse_args()
    args.stmts = (args.stmts.lower() == 'true')
    args.trigger = (args.trigger.lower() == 'true')
    args.full = (args.full.lower()=='true')
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
        else:
            logger.info(f"Loading small model from {args.s_model_name_or_path}")
            s_tokenizer = AutoTokenizer.from_pretrained(args.s_model_name_or_path)
            s_model = AutoModelForCausalLM.from_pretrained(args.s_model_name_or_path).to(args.device)
            s_model.eval()
            logger.info("Small model loaded.")
            
        s_tokenizer = AutoTokenizer.from_pretrained(args.s_model_name_or_path)
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
    
    ## 加载parser
    PY_LANGUAGE = Language(tspython.language())
    parser = Parser(PY_LANGUAGE)

    step = "step7-more"
    if s_tokenizer.pad_token is None:
        s_tokenizer.pad_token = s_tokenizer.eos_token
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
    
    
    special_token_ids = {
        "prefix_id": s_tokenizer.convert_tokens_to_ids("<PREFIX>"),
        "suffix_id": s_tokenizer.convert_tokens_to_ids("<SUFFIX>"),
        "middle_id": s_tokenizer.convert_tokens_to_ids("<MIDDLE>"),
        "eos_id": s_tokenizer.convert_tokens_to_ids("<EOS>")
    }
    for name,examples in tqdm(all_eval_examples.items(),desc="Formatting"):
        print(name)
        if name=='repoeval_func' or name=='ours_func':
            examples = [example for x in examples for example in x]
            print(len(examples))
        for example in examples:
            example.input_ids,example.relevent_str,example.prefix_str,example.suffix_str = format_example(args,s_tokenizer,special_token_ids,example)
            
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
            l_model=l_model,
            l_model_type=args.large_model_type, # e.g., 'deepseek', 'qwen'
            l_tokenizer=l_tokenizer,
            device=args.device,
            lang=args.lang,
            s_model=s_model,
            s_model_type="opc", # e.g., 'opc', 'llama'
            s_tokenizer=s_tokenizer,
            parser=parser,
            without_sp=args.without_sp
        )

    # 创建时间统计字典和结果目录
    time_stats = {}
    os.makedirs(args.output_dir, exist_ok=True)
    flash = 5
   
    for key, examples in tqdm(all_eval_examples.items(), desc="Generating"):
        if decoder.task!=key:
            decoder.set_task(key)
            print(f"Switching to {key} task")
        if key=='repoeval_func' or key=='ours_func':
            examples = [example for x in examples for example in x]
        time_stats[key] = {}
        # 开始计时
        # start_time = time.time()
        i = 0
        results = []
        logger.info("Evaluating on {} dataset".format(key))
        save_path = f"{args.output_dir}/result/{key}/prediction.jsonl"
        if not os.path.exists(save_path):
            for example in tqdm(examples):
                sample_start_time = time.time()
                if args.one_model:
                    generated_code, small_output = decoder.generate(
                        input_ids=example.input_ids,
                        max_length=args.max_input_length,
                        prefix=example.prefix_str,
                        suffix=example.suffix_str,
                        relevent_str=example.relevent_str,
                        key=key,
                        ground_truth=example.middle,
                        prompt=example.prefix
                    )
                else:
                    generated_code, small_output = decoder.generate(
                        input_ids=example.input_ids,
                        max_length=args.max_input_length,
                        prefix=example.prefix_str,
                        suffix=example.suffix_str,
                        relevent_str=example.relevent_str,
                        key=key,
                        ground_truth=example.middle,
                        trigger_point_idx=example.trigger_point_idx if args.trigger else None,
                        prompt=example.prefix
                    )
                # print(example.task_id)
                # print("groundtruth:",example.middle)
                # print("trigger input",example.middle[:example.trigger_point_idx])
                # print("generated code",generated_code)
                sample_end_time = time.time()
                if args.trigger:
                    if small_output!=None:
                        generated_code = example.middle[:example.trigger_point_idx]+generated_code
                    else:
                        generated_code = generated_code
                else:
                    generated_code = generated_code
                    
                results.append({
                    "id": example.task_id,
                    "prefix": example.prefix,
                    "suffix": example.suffix,
                    "generated": generated_code,
                    "ground_truth":example.middle,
                    "time": sample_end_time - sample_start_time,
                    "small_output":small_output,
                    'original_small_ouput':example.small_pred
                })
                i+=1
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
                        'ground_truth':result['ground_truth'],'prefix':result['prefix'],'suffix':result['suffix'],'original_small_ouput':result['original_small_ouput']}) + "\n")
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

def generate_dual_story(prompt, model1, model2):
    # 分别生成并立即保存结果
    story1 = model1.generate(prompt).text
    # 清理第一个生成的中间结果
    torch.cuda.empty_cache()
    
    story2 = model2.generate(prompt).text
    # 清理第二个生成的中间结果
    torch.cuda.empty_cache()
    
    return story1, story2

if __name__ == "__main__":
    main()