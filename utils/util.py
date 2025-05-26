import pandas as pd
import tokenize
import io
import re
import os
import rank_bm25
import pickle
from nltk.tokenize import word_tokenize
from typing import Tuple, List, Dict
from transformers import PreTrainedTokenizer

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_ids:List[int],
                 attention_mask:List[int],
                 target_ids:List[int]
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.target_ids = target_ids

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
                relevant_codes:List["CodeBlock"], small_pred:str=None, correct:bool=None, \
                full_line_relevant_stmts:List["CodeBlock"]=[], full_line_relevant_codes:List["CodeBlock"]=[],
                small_relevant_stmts:List["CodeBlock"]=[], small_relevant_codes:List["CodeBlock"]=[], \
                small_relevant_local_stmts:List["CodeBlock"]=[]) -> None:
        self.task_id = task_id
        self.prefix = prefix
        self.suffix = suffix
        self.middle = middle
        self.relevant_codes = relevant_codes
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


def load_dataset(datasetname:str, tokenizer_name:str, k:int) -> List[Example]:
    """
    Loads a dataset.
    :param datasetname: The name of the dataset to load.
    :return: The loaded dataset.
    """
    file_name = f"{datasetname}-{tokenizer_name}-{k}.pkl"
    with open(f"preprocessed/{file_name}", 'rb') as f:
        dataset = pickle.load(f)
    
    return dataset

def load_dataset_from_path(path:str) -> List[Example]:
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    
    return dataset

def split_sentence(code:str) -> List[str]:
    return word_tokenize(code)

def bm25_retrieve(query_str:str, candidate_str:List[str], k:int):
    if k == 0 or len(candidate_str) == 0:
        return []
    # TODO: 将检索使用的token数量设置为一个参数
    tokenized_corpus = [split_sentence(doc) for doc in candidate_str]
    bm25_model = rank_bm25.BM25Okapi(tokenized_corpus)
    query = split_sentence(query_str)
    doc_scores = bm25_model.get_scores(query)
    return doc_scores

def relevant_contexts(small_related_code:List[CodeBlock], repo_related_codes:List[CodeBlock], tokenizer:PreTrainedTokenizer, cross_file_budget:int, small_repo_percent:float=0.8) -> Dict[str, List[int]]:
    filter_small_codeblocks = []
    for x in small_related_code:
        file_path = x.file_path
        code_content = x.code_content
        filter_small_codeblocks.append(f"{code_content}")

    filter_repo_codeblocks = []

    for x in repo_related_codes:
        file_path = x.file_path
        code_content = x.code_content
        if file_path != "" and file_path != "Unknown":
            filter_repo_codeblocks.append(f"#{file_path}\n{code_content}" if code_content.endswith("\n") else f"#{file_path}\n{code_content}\n")
        else:
            break
    
    repo_content = {
        "input_ids": [],
        "attention_mask": []
    }
    small_input_content = {
        "input_ids": [],
        "attention_mask": []
    }
    
    if len(filter_repo_codeblocks) > 0:
        related_tokenized_repo_result = tokenizer(filter_repo_codeblocks, add_special_tokens=False)
    else:
        # 初始化为空字典而不是None，避免后续访问属性时出错
        related_tokenized_repo_result = {"input_ids": [[]], "attention_mask": [[]]}
    if len(filter_small_codeblocks) > 0:
        related_tokenized_small_result = tokenizer(filter_small_codeblocks, add_special_tokens=False)
    else:
        # 初始化为空字典而不是None，避免后续访问属性时出错
        related_tokenized_small_result = {"input_ids": [[]], "attention_mask": [[]]}
    
    # 检查两个结果是否都为空（即都没有实际内容）
    if len(related_tokenized_repo_result["input_ids"][0]) == 0 and len(related_tokenized_small_result["input_ids"][0]) == 0:
        return repo_content, small_input_content
    elif len(related_tokenized_repo_result["input_ids"][0]) == 0:
        small_budget = cross_file_budget
        repo_budget = 0
    elif len(related_tokenized_small_result["input_ids"][0]) == 0:
        small_budget = 0
        repo_budget = cross_file_budget
    else:
        small_budget = int((1 - small_repo_percent) * cross_file_budget)
        repo_budget = int(small_repo_percent * cross_file_budget)
    
    repo_content["input_ids"] =  related_tokenized_small_result["input_ids"][0][:small_budget]+related_tokenized_repo_result["input_ids"][0][:repo_budget]
    repo_content["attention_mask"] = related_tokenized_small_result["attention_mask"][0][:small_budget]+related_tokenized_repo_result["attention_mask"][0][:repo_budget]
    
    small_input_content["input_ids"] =  related_tokenized_repo_result["input_ids"][0][:cross_file_budget]
    small_input_content["attention_mask"] = related_tokenized_repo_result["attention_mask"][0][:cross_file_budget]
    ## repocontent 是包含小模型检索的内容，small——input就是只有根据上下文检索的
    return repo_content, small_input_content