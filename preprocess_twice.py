from utils.util import bm25_retrieve
from utils.eval_util import process_example_inline
from tqdm import tqdm
from typing import List
from multiprocessing import Pool
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tiktoken
import rank_bm25
import argparse
import pandas as pd
import numpy as np
import pickle
import os
import json
import io
import tokenize

# Global variables for multiprocessing
global_generated_completion = None
global_args = None
global_encoder = None
global_parser = None

TRIGGER_POINT = [
    ".",
    "(",
    "=",
    ",",
    "/",
    "-",
    "[",
    "<",
    "{",
    "if",
    ">",
    "return",
    "*",
    "+",
    "in",
    "or",
    "==",
    ";",
    "and",
    "is",
    "&",
    "for",
    "%",
    "|",
    "else",
    "not",
    "with",
    "while",
    "await",
    "!=",
    "+=",
    "del",
    "**",
    "^",
    "assert",
    "~",
    "except",
    ">=",
    ">>",
    "<<",
    "global",
    "raise",
    "<=",
    "elif",
    "yield",
    "-=",
    "lambda"
]

def tiktoken_split_sentence(code:str):
    return [global_encoder.decode([token_id]) for token_id in global_encoder.encode(code)]

def tiktoken_bm25_retrieve(query_str:str, candidate_str:List[str], k:int):
    if k == 0 or len(candidate_str) == 0:
        return []
    # TODO: 将检索使用的token数量设置为一个参数
    tokenized_corpus = [tiktoken_split_sentence(doc) for doc in candidate_str]
    bm25_model = rank_bm25.BM25Okapi(tokenized_corpus)
    query = tiktoken_split_sentence(query_str)
    doc_scores = bm25_model.get_scores(query)
    return doc_scores

def find_min_end(s: str, lst: List[str]):
    min_end = None
    for sub in lst:
        if not sub:  # 跳过空字符串
            continue
        if sub in s:
            start = s.find(sub)
            end = start + len(sub) - 1
            if (min_end is None) or (end < min_end):
                min_end = end
    return min_end if min_end is not None else None

class Example:
    def __init__(self, task_id:str, prefix:str, suffix:str, middle:str, \
                relevant_codes:List["CodeBlock"], relevant_code_scores:List[float], small_pred:str, correct:bool, \
                full_line_relevant_stmts:List["CodeBlock"], full_line_relevant_codes:List["CodeBlock"],
                small_relevant_stmts:List["CodeBlock"], small_relevant_codes:List["CodeBlock"], \
                small_relevant_local_stmts:List["CodeBlock"]) -> None:
        self.task_id = task_id
        self.prefix = prefix
        self.suffix = suffix
        self.middle = middle
        self.relevant_codes = relevant_codes
        self.relevant_code_scores = relevant_code_scores
        self.small_pred = small_pred
        self.correct = correct
        self.small_relevant_stmts = small_relevant_stmts
        self.small_relevant_codes = small_relevant_codes
        self.full_line_relevant_stmts = full_line_relevant_stmts
        self.full_line_relevant_codes = full_line_relevant_codes
        self.small_relevant_local_stmts = small_relevant_local_stmts
        self.trigger_point_idx = find_min_end(self.middle, TRIGGER_POINT)
    

class CodeBlock(object):
    def __init__(self, file_path:str, code_content:str):
        """
        Represents a block of code.
        :param file_path: The path to the code file.
        :param code_content: The content of the code block.
        """
        self.file_path:str = file_path
        self.code_content:str = code_content
            
    def __str__(self):
        return f"#{self.file_path}\n{self.code_content}"
    
    def __eq__(self, value):
        assert type(value) == CodeBlock
        if self.file_path == value.file_path and self.code_content == value.code_content:
            return True
        else:
            return False

def split_into_smaller_blocks(code_block:CodeBlock, windows_length, step) -> List[CodeBlock]:
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

def split_into_stmt(code:str):
    stack = []
    statements = []
    line_count = 0
    tokens = tokenize.generate_tokens(io.StringIO(code).readline)
    raw_lines = code.splitlines(keepends=True)
    
    try:
        for token_type, string, start, _, _ in tokens:
            if token_type == tokenize.OP:
                if string == '{' or string == '[' or string == '(':
                    stack.append(string)
                elif string == '}' or string == ']' or string == ')':
                    stack.pop()
                        
            if token_type == tokenize.NL and len(stack) == 0:
                statements.append(raw_lines[start[0] - 1])
                line_count = start[0]
            
            elif token_type == tokenize.NEWLINE:
                statements.append("".join([raw_lines[i - 1] for i in range(line_count+1, start[0]+1)]))
                line_count = start[0]
    except:
        return []
    return statements

def init_pool(args):
    """Initialize global variables for each pool worker"""
    global global_generated_completion, global_args, global_encoder, global_parser
    global_args = args
    global_generated_completion = {}
    global_encoder = tiktoken.get_encoding("cl100k_base")
    with open(args.generated_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            global_generated_completion[js["task_id"]] = js["pred"]
    PY_LANGUAGE = Language(tspython.language())
    global_parser = Parser(PY_LANGUAGE)

def process_item(item):
    """Process single item with global variables"""
    task_id, path, left_context, right_context, crossfile_context, groundtruth = item
    left_context:str
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
        suffix_part = []
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
    retrieved_codeblocks_scores = [scores[idx] for idx in sorted_indices]
    
    completion = global_generated_completion[task_id]
    _, correct = process_example_inline("python", global_parser, completion, groundtruth, left_context)
    
    # 根据小模型输出的内容进行检索
    statement_blocks = []
    for file in cross_files:
        statement_blocks.extend([CodeBlock(file.file_path, stmt) for stmt in split_into_stmt(file.code_content)])
    completion_stmt_scores = bm25_retrieve(completion, 
                            [sb.code_content for sb in statement_blocks if sb.code_content.strip()],
                            global_args.relevant_code_num)
    completion_stmt_sorted_indices = np.argsort(completion_stmt_scores)[::-1][:global_args.relevant_code_num]
    completion_retrieved_stmts = [statement_blocks[idx] for idx in completion_stmt_sorted_indices]
    
    completion_codeblock_scores = bm25_retrieve(
        completion, [cb.code_content for cb in code_blocks], global_args.relevant_code_num
    )
    completion_codeblock_sorted_indices = np.argsort(completion_codeblock_scores)[::-1][:global_args.relevant_code_num]
    completion_retrieved_codeblocks = [code_blocks[idx] for idx in completion_codeblock_sorted_indices]
    
    # full_stmt_blocks = [CodeBlock(path, stmt) for stmt in split_into_stmt(left_context + groundtruth + "\n" + right_context)]
    # assert full_stmt_blocks
    # local_stmt_blocks = []
    # left_context_line_count = left_context.count("\n")
    # cur_line_count = 0
    # for idx, cb in enumerate(full_stmt_blocks):
    #     cur_line_count += cb.code_content.count("\n")
    #     if cur_line_count > left_context_line_count:
    #         break
    # local_stmt_blocks += full_stmt_blocks[:idx]
    # # 确保groundtruth没有被纳入local_stmt_blocks
    # assert groundtruth in full_stmt_blocks[idx].code_content
    # if not global_args.only_prefix:
    #     local_stmt_blocks += full_stmt_blocks[idx+1:]
    
    # if local_stmt_blocks:
    #     completion_local_stmt_scores = bm25_retrieve(
    #         completion,
    #         [sb.code_content for sb in local_stmt_blocks if sb.code_content.strip()],
    #         global_args.relevant_code_num
    #     )
    # else:
    #     completion_local_stmt_scores = None
    # completion_local_stmt_sorted_indices = np.argsort(completion_local_stmt_scores)[::-1][:global_args.relevant_code_num]
    # completion_retrieved_local_stmts = [local_stmt_blocks[idx] for idx in completion_local_stmt_sorted_indices]
    
    if left_context.endswith("\n"):
        full_line_relevant_stmts = None
        full_line_relevant_codes = None
    else:
        full_line_query = left_context.splitlines()[-1] + completion
        full_line_stmt_scores = bm25_retrieve(
            full_line_query,
            [sb.code_content for sb in statement_blocks if sb.code_content.strip()],
            global_args.relevant_code_num
        )
        full_line_stmt_sorted_indices = np.argsort(full_line_stmt_scores)[::-1][:global_args.relevant_code_num]
        full_line_relevant_stmts = [statement_blocks[idx] for idx in full_line_stmt_sorted_indices]
        
        full_line_codeblock_scores = bm25_retrieve(
            full_line_query,
            [cb.code_content for cb in code_blocks],
            global_args.relevant_code_num
        )
        full_line_codeblock_sorted_indices = np.argsort(full_line_codeblock_scores)[::-1][:global_args.relevant_code_num]
        full_line_relevant_codes = [code_blocks[idx] for idx in full_line_codeblock_sorted_indices]
    
    if global_args.only_prefix:
        right_context = ""
        
    if global_args.dataset_name == "repoeval_line" or global_args.dataset_name == "repoeval_api" or \
            global_args.dataset_name == "repoeval_line_only_prefix" or global_args.dataset_name == "repoeval_api_only_prefix":
        left_context = left_context + "\n"
    
    return Example(
        task_id=task_id, 
        prefix=left_context, 
        suffix=right_context, 
        middle=groundtruth, 
        relevant_codes=retrieved_codeblocks, 
        relevant_code_scores=retrieved_codeblocks_scores,
        small_pred=completion,
        correct=correct,
        small_relevant_stmts=completion_retrieved_stmts,
        small_relevant_codes=completion_retrieved_codeblocks,
        full_line_relevant_stmts = full_line_relevant_stmts,
        full_line_relevant_codes = full_line_relevant_codes,
        small_relevant_local_stmts = None
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--relevant_code_num", type=int, default=5)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--generated_file", type=str, required=True)
    parser.add_argument("--func_level", action="store_true")
    parser.add_argument("--only_prefix", action="store_true")
    parser.add_argument("--step", type=int, default=7)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    if not args.only_prefix:
        assert "only-prefix" not in args.generated_file
        assert "only_prefix" not in args.generated_file
        assert "only-prefix" not in args.dataset_path
        assert "only_prefix" not in args.dataset_path
    
    assert not args.func_level
    
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

    func = process_item
    
    # Create process pool
    with Pool(processes=args.workers,
             initializer=init_pool,
             initargs=(args, )) as pool:
        
        results = list(tqdm(pool.imap(func, items, chunksize=10),
                      total=len(items),
                      desc=f"Processing {args.dataset_name}"))
    
    # Filter failed items and save
    if args.func_level:
        processed_dataset = [e for examples in results for e in examples]
    else:
        processed_dataset = [r for r in results if r is not None]
    
    output_dir = args.output_dir
    
    correct = [1 for i in processed_dataset if i.correct]
    print(f"{args.dataset_name} : {len(correct) / len(processed_dataset)}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, 
                              f"{args.dataset_name}-{args.relevant_code_num}.pkl")
    
    with open(output_path, 'wb') as f:
        pickle.dump(processed_dataset, f)

if __name__ == "__main__":
    main()