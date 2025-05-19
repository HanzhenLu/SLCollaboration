import pandas as pd
import tokenize
import io
import re
import os
import rank_bm25
import pickle
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
                relevant_codes:List["CodeBlock"], small_pred:str, correct:bool, \
                full_line_relevant_stmts:List["CodeBlock"], full_line_relevant_codes:List["CodeBlock"],
                small_relevant_stmts:List["CodeBlock"], small_relevant_codes:List["CodeBlock"], \
                small_relevant_local_stmts:List["CodeBlock"]) -> None:
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
    
def label_line(code:str) -> List[Tuple[List[int], bool]]:
    stack = []
    line_map = []
    line_count = 0
    tokens = tokenize.generate_tokens(io.StringIO(code).readline)
    
    for token_type, string, start, _, _ in tokens:
        # OP
        if token_type == 54:
            if string == '{' or string == '[' or string == '(':
                stack.append(string)
            elif string == '}' or string == ']' or string == ')':
                stack.pop()
                    
        # NL
        if token_type == 61 and len(stack) == 0:
            line_map.append(([start[0] - 1], False))
            line_count = start[0]
        
        # NEWLINE
        elif token_type == 4:
            line_map.append(([i - 1 for i in range(line_count+1, start[0]+1)], True))
            line_count = start[0]
    
    return line_map

def split_into_smaller_blocks(code_block:CodeBlock, enable_fixed_block:bool) -> List[CodeBlock]:
    """
    Split large blocks of code into smaller ones, each containing no more than 12 non-empty lines.
    """
    smaller_blocks = []

    # 每15行划分一个block
    if enable_fixed_block:
        lines = [line for line in code_block.code_content.split('\n') if line.strip() != '']
        for i in range(0, min(len(lines),5000), 8):
            start_line_offset = i
            end_line_offset = min(i + 15, len(lines))
            block_content = '\n'.join(lines[start_line_offset:end_line_offset])
            smaller_blocks.append(CodeBlock(code_block.file_path, 
                                            block_content))

    else:
        # Split the code by spaces, then reassemble it into blocks.
        mini_blocks = []
        current_block = [] 
        for line in code_block.code_content.splitlines(): 
            if line.strip() == '':  
                if current_block: 
                    mini_blocks.append(current_block)
                    current_block = []
            else:
                current_block.append(line)
        if current_block: 
            mini_blocks.append(current_block)

        # 超过12行的block划分成多个block
        max_len = 15
        temp_mini_blocks = []
        for mini_block in mini_blocks:
            if len(mini_block) > max_len:
                for idx in range(0, len(mini_block), max_len):
                    temp_mini_blocks.append(mini_block[idx: idx+max_len])
            else:
                temp_mini_blocks.append(mini_block)
        mini_blocks = temp_mini_blocks

        current_content = []
        total_lines = 0  
        for block in mini_blocks:
            if total_lines >= 5000:  
                break  
            if len(current_content) + len(block) <= max_len:  
                current_content.extend(block)
                total_lines += len(block)  
            else:  
                if current_content:  
                    smaller_blocks.append(CodeBlock(code_block.file_path, 
                                                    '\n'.join(current_content)))
                current_content = block  
                total_lines += len(block)  
        if current_content:  
            smaller_blocks.append(CodeBlock(code_block.file_path, 
                                            '\n'.join(current_content)))
        
    return smaller_blocks

def split_word(word:str) -> List[str]:
    words = []
    
    if len(word) <= 1:
        return word

    word_parts = re.split('[^0-9a-zA-Z]', word)
    for part in word_parts:
        part_len = len(part)
        if part_len == 1:
            words.append(part)
            continue
        word = ''
        for index, char in enumerate(part):
            # condition : level|A
            if index == part_len - 1 and char.isupper() and part[index-1].islower():
                if word != '':
                    words.append(word)
                words.append(char)
                word = ''
                
            elif(index != 0 and index != part_len - 1 and char.isupper()):
                # condition 1 : FIRST|Name
                # condition 2 : first|Name
                condition1 = part[index-1].isalpha() and part[index+1].islower()
                condition2 = part[index-1].islower() and part[index+1].isalpha()
                if condition1 or condition2:
                    if word != '':
                        words.append(word)
                    word = char
                else:
                    word += char
            
            else:
                word += char
        
        if word != '':
            words.append(word)
            
    return [word.lower() for word in words]

def bm25_retrieve(query_str:str, candidate_str:List[str], tokenizer:PreTrainedTokenizer, k:int):
    if k == 0 or len(candidate_str) == 0:
        return []
    # TODO: 将检索使用的token数量设置为一个参数
    tokenized_corpus = [tokenizer.tokenize(doc) for doc in candidate_str]
    bm25_model = rank_bm25.BM25Okapi(tokenized_corpus)
    query = tokenizer.tokenize(query_str)
    doc_scores = bm25_model.get_scores(query)
    return doc_scores

def cross_file_contexts(related_codes:List[CodeBlock], tokenizer:PreTrainedTokenizer, cross_file_budget:int) -> Dict[str, List[int]]:
    filter_codeblocks = []
    for x in related_codes:
        file_path = x.file_path
        code_content = x.code_content
        if file_path != "" and file_path != "Unknown":
            filter_codeblocks.append(f"#{file_path}\n{code_content}" if code_content.endswith("\n") else f"#{file_path}\n{code_content}\n")
        else:
            break
    
    repo_content = {
        "input_ids": [],
        "attention_mask": []
    }
    
    if len(filter_codeblocks) > 0:
        related_tokenized_result = tokenizer(filter_codeblocks, add_special_tokens=False)
    else:
        return repo_content
    
    special_tokens = tokenizer.all_special_tokens
    if "<RETRIEVAL_START>" in special_tokens and "<RETRIEVAL_END>" in special_tokens:
        repo_content["input_ids"] = [tokenizer.convert_tokens_to_ids("<RETRIEVAL_START>")] + \
            related_tokenized_result["input_ids"][0][:cross_file_budget - 2] + [tokenizer.convert_tokens_to_ids("<RETRIEVAL_END>")]
        repo_content["attention_mask"] = [1] + related_tokenized_result["attention_mask"][0][:cross_file_budget - 2] + [1]
    else:
        repo_content["input_ids"] = related_tokenized_result["input_ids"][0][:cross_file_budget]
        repo_content["attention_mask"] = related_tokenized_result["attention_mask"][0][:cross_file_budget]
    
    return repo_content

def relevent_contexts(small_related_code:List[CodeBlock], repo_related_codes:List[CodeBlock], tokenizer:PreTrainedTokenizer, cross_file_budget:int, small_repo_percent:float=0.8) -> Dict[str, List[int]]:
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