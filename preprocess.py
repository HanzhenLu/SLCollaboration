from utils.util import (bm25_retrieve,
                        CodeBlock, Example)
from tqdm import tqdm
import argparse
import pandas as pd
import numpy as np
import pickle
import os
import json
from multiprocessing import Pool

# Global variables for multiprocessing
global_args = None

def split_into_smaller_blocks(code_block:CodeBlock, windows_length, step):
    """
    Split large blocks of code into smaller ones, each containing no more than 12 non-empty lines.
    """
    smaller_blocks = []

    lines = [line for line in code_block.code_content.split('\n') if line.strip() != '']
    for i in range(0, min(len(lines),5000), step):
        start_line_offset = i
        end_line_offset = min(i + windows_length, len(lines))
        block_content = '\n'.join(lines[start_line_offset:end_line_offset])
        smaller_blocks.append(CodeBlock(code_block.file_path, 
                                        block_content))
        
    return smaller_blocks

def init_pool(args):
    """Initialize global variables for each pool worker"""
    global global_args
    global_args = args

def process_item(item):
    """Process single item with global variables"""
    try:
        task_id, _, left_context, right_context, crossfile_context, groundtruth = item
        
        # Process crossfile context
        cross_files = crossfile_context if len(crossfile_context) > 0 else [
            {'path': "", "text": "Don't need cross file context for completion"}]
        cross_files = [CodeBlock(x["path"], x["text"]) for x in cross_files]
        
        code_blocks = []
        for file in cross_files:
            code_blocks.extend(split_into_smaller_blocks(file, 15, global_args.step))
        
        # Process context windows
        if global_args.only_prefix:
            prefix_line_clean = [line for line in left_context.split('\n') if line.strip()]
            prefix_part = prefix_line_clean[-15:] if len(prefix_line_clean) >= 15 else prefix_line_clean
            query_str = "\n".join(prefix_part)
            right_context = ""
        else:
            prefix_line_clean = [line for line in left_context.split('\n') if line.strip()]
            prefix_part = prefix_line_clean[-8:] if len(prefix_line_clean) >= 8 else prefix_line_clean
            remaining = 15 - len(prefix_part)
            suffix_line_clean = [line for line in right_context.split('\n') if line.strip()]
            suffix_part = suffix_line_clean[:remaining]
            query_str = "\n".join(prefix_part + suffix_part)
        
        # BM25 retrieval
        scores = bm25_retrieve(query_str, 
                            [cb.code_content for cb in code_blocks],
                            global_args.relevant_code_num)
        sorted_indices = np.argsort(scores)[::-1][:global_args.relevant_code_num]
        retrieved_codeblocks = [code_blocks[idx] for idx in sorted_indices]
        
        # Create Example object
        if global_args.dataset_name == "repoeval_line" or global_args.dataset_name == "repoeval_api" or \
            global_args.dataset_name == "repoeval_line_only_prefix" or global_args.dataset_name == "repoeval_api_only_prefix":
            return Example(task_id, left_context+"\n", right_context, groundtruth, retrieved_codeblocks)
        return Example(task_id, left_context, right_context, groundtruth, retrieved_codeblocks)
    except Exception as e:
        print(f"Error processing {task_id}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--relevant_code_num", type=int, default=5)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only_prefix", action="store_true")
    parser.add_argument("--step", type=int, default=7)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    # Load dataset
    dataset_paths = args.dataset_path.split(",") if "," in args.dataset_path else [args.dataset_path]
    datasets = []
    for path in dataset_paths:
        if path.endswith(".json"):
            try:
                datasets.append(pd.read_json(path))
            except ValueError as e:
                datasets.append(pd.read_json(path, lines=True))
        elif path.endswith(".parquet"):
            datasets.append(pd.read_parquet(path))
        else:
            raise RuntimeError("Unsupported file type")
    dataset = pd.concat(datasets)

    # Prepare items for multiprocessing
    items = dataset[["task_id", "path", "left_context", "right_context", 
                   "crossfile_context", "groundtruth"]].values.tolist()
    
    # Create process pool
    with Pool(processes=args.workers,
            initializer=init_pool,
            initargs=(args,)) as pool:
        
        results = list(tqdm(pool.imap(process_item, items, chunksize=10),
                    total=len(items),
                    desc=f"Processing {args.dataset_name}"))
    
    processed_dataset = [r for r in results if r is not None]
    
    output_dir = args.output_dir
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, 
                              f"{args.dataset_name}-{args.relevant_code_num}.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(processed_dataset, f)

if __name__ == "__main__":
    main()