import torch
import time
from line_profiler import profile
from typing import List
from transformers import StoppingCriteria, StoppingCriteriaList
from utils.eval_util import is_parse_valid

@profile
def greedy_generate(model, tokenizer, input_ids, max_new_tokens, past_key_values, stopping_criteria):
    model.eval()

    generated = input_ids.clone()
    next_token = input_ids
    eos_token_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            # 只输入上一个 token，同时传入 past_key_values
            outputs = model(next_token, use_cache=True, past_key_values=past_key_values)
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            # 贪婪选择
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated = torch.cat([generated, next_token], dim=-1)

            # 停止条件
            if next_token.item() == eos_token_id or stopping_criteria(generated, next_token_logits):
                break

    return tokenizer.decode(generated[0], skip_special_tokens=True)


class PythonStatementStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt, input_ids_len, parser, NL_list:List[int]):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.input_ids_len = input_ids_len  # 原始 prompt 的 token 长度
        self.parser = parser
        self.NL_list = NL_list

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 检查是否出现换行
        if not input_ids[0][-1].detach().cpu().numpy() in self.NL_list:
            return False
        
        # 获取新生成的部分（不包含 prompt）
        generated_ids = input_ids[0, self.input_ids_len:]
        new_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        if is_parse_valid(self.parser, self.prompt + new_text):
            return True  # 满足停止条件
        else:
            return False  # 继续生成
        
class PythonLineStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, input_ids_len, NL_list:List[int]):
        self.tokenizer = tokenizer
        self.input_ids_len = input_ids_len  # 原始 prompt 的 token 长度
        self.NL_list = NL_list

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 检查是否出现换行
        if not input_ids[0][-1].detach().cpu().numpy() in self.NL_list:
            return False
        
        return True

class SPDecoding:
    def __init__(self, model, model_type, tokenizer, max_new_tokens, device, parser, without_sp=False):
        self.model = model
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.without_sp = without_sp
        
        self.parser = parser
        self.task = None
        self.token_mapping = {}
        if 'qwen' == model_type:
            self.token_mapping["prefix"] = "<|fim_prefix|>"
            self.token_mapping["middle"] = "<|fim_middle|>"
            self.token_mapping["suffix"] = "<|fim_suffix|>"
        elif 'deepseek' == model_type:
            self.token_mapping["prefix"] = '<｜fim▁begin｜>'
            self.token_mapping["middle"] = '<｜fim▁hole｜>'
            self.token_mapping["suffix"] = '<｜fim▁end｜>'
        elif "starcoder" == model_type:
            self.token_mapping["prefix"] = "<fim_prefix>"
            self.token_mapping["middle"] = "<fim_middle>"
            self.token_mapping["suffix"] = "<fim_suffix>"
        elif "opc" == model_type:
            self.token_mapping["prefix"] = "<PREFIX>"
            self.token_mapping["middle"] = "<MIDDLE>"
            self.token_mapping["suffix"] = "<SUFFIX>"
            self.tokenizer.eos_token_id = 2
            self.tokenizer.eos_token = "<EOS>"
        elif "llama" == model_type:
            self.token_mapping["prefix"] = "▁<PRE>"
            self.token_mapping["middle"] = "▁<MID>"
            self.token_mapping["suffix"] = "▁<SUF>"

        self.NL_list = [id for _, id in tokenizer.vocab.items() if tokenizer.decode(id).endswith("\n")]
    
    def encode_infilling(self, s:str):
        """专门给CodeLlama使用的特殊encode"""
        return self.tokenizer.encode("☺" + s)[3:]

    def decode_infilling(self, t:List[int]):
        """专门给CodeLlama使用的特殊decode"""
        return self.tokenizer.decode([self.tokenizer.convert_tokens_to_ids("☺")] + t, skip_special_tokens=True)[1:]
        
    def set_task(self,task):
        self.task = task
    
    @profile
    def generate(self, prefix="", suffix="", relevant_str="", ground_truth="", trigger_point_idx=None, **kwargs):
        """生成代码"""
        small_output = kwargs["small_output"]
        
        if trigger_point_idx is not None:
            trigger_str = ground_truth[:trigger_point_idx+1]
            prefix += trigger_str
            if small_output.startswith(trigger_str):
                small_output = small_output[trigger_point_idx+1:]
            else:
                small_output = ""
            flag = "trigger"
        elif self.without_sp:
            flag = "Not_veri"
        else:
            flag = "verified"
        
        # 准备用于投机采样的输入和正常的输入
        if self.model_type in ["qwen", "starcoder"]:
            context = f"{self.token_mapping['prefix']}{relevant_str}{prefix}{self.token_mapping['suffix']}{suffix}{self.token_mapping['middle']}{small_output}"
            origin_context = f"{self.token_mapping['prefix']}{relevant_str}{prefix}{self.token_mapping['suffix']}{suffix}{self.token_mapping['middle']}"
        elif self.model_type == "deepseek":
            context = f"{self.token_mapping['prefix']}{relevant_str}{prefix}{self.token_mapping['middle']}{suffix}{self.token_mapping['suffix']}{small_output}"
            origin_context = f"{self.token_mapping['prefix']}{relevant_str}{prefix}{self.token_mapping['middle']}{suffix}{self.token_mapping['suffix']}"
        elif self.model_type == "opc":
            context = f"{self.token_mapping['suffix']}{suffix}{self.token_mapping['prefix']}{relevant_str}{prefix}{self.token_mapping['middle']}{small_output}"
            origin_context = f"{self.token_mapping['suffix']}{suffix}{self.token_mapping['prefix']}{relevant_str}{prefix}{self.token_mapping['middle']}"

        if self.model_type == "opc":
            large_input_ids = self.tokenizer(context, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            origin_large_input_ids = self.tokenizer(origin_context, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        elif self.model_type != "llama":
            large_input_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(self.device)
            origin_large_input_ids = self.tokenizer(origin_context, return_tensors="pt").input_ids.to(self.device)

        # CodeLlama 会在句首插入空格，导致token不一致，因此需要特殊处理
        if self.model_type == "llama":
            large_input_ids = [
                [self.tokenizer.bos_token_id, self.tokenizer.prefix_id]
                + self.tokenizer.encode(f"{relevant_str}{prefix}", add_special_tokens=False)
                + [self.tokenizer.suffix_id] + self.encode_infilling(suffix) + [self.tokenizer.middle_id]
                + self.encode_infilling(small_output)
            ]
            large_input_ids = torch.tensor(large_input_ids, dtype=torch.long).to(self.device)
            
            origin_large_input_ids = [
                [self.tokenizer.bos_token_id, self.tokenizer.prefix_id]
                + self.tokenizer.encode(f"{relevant_str}{prefix}", add_special_tokens=False)
                + [self.tokenizer.suffix_id] + self.encode_infilling(suffix) + [self.tokenizer.middle_id]
            ]
            origin_large_input_ids = torch.tensor(origin_large_input_ids, dtype=torch.long).to(self.device)
            
            small_output_tokens = self.encode_infilling(small_output)
            small_output_tokens = torch.tensor(small_output_tokens, dtype=torch.long).to(self.device)
        else:
            small_output_tokens = self.tokenizer(small_output, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)[0]
        small_output_index = origin_large_input_ids.shape[1]
        
        # 开始生成
        with torch.no_grad():
            if flag == 'verified' or flag == "trigger":
                if len(small_output_tokens) == 0:
                    large_input_ids = origin_large_input_ids
                    mismatch_pos = "NotValidate"
                    max_new_tokens = self.max_new_tokens
                    flag = "Not_veri"
                    mismatch = None
                    early_stop_time = None
                else:
                    outputs = self.model(large_input_ids, use_cache=True)
                    logits = outputs.logits
                    kv_cache = outputs.past_key_values
                    small_end_index = small_output_index + len(small_output_tokens)

                    # 获取每个位置的预测token
                    pred_tokens = torch.argmax(logits[0, small_output_index-1:small_end_index], dim=-1)
                    
                    mismatch = (pred_tokens[:-1] != small_output_tokens).nonzero(as_tuple=True)
                    early_stop_time = time.time()
                    if mismatch[0].numel() > 0:
                        mismatch_pos = mismatch[0][0].item()
                        large_input_ids = pred_tokens[mismatch_pos:mismatch_pos+1].unsqueeze(0)
                        kv_cache = [
                            (k[:, :, :len(origin_large_input_ids[0]) + mismatch_pos], v[:, :, :len(origin_large_input_ids[0]) + mismatch_pos]) for k, v in kv_cache
                        ]
                    else:
                        mismatch_pos = len(small_output_tokens)
                        kv_cache = [
                            (k[:, :, :len(large_input_ids[0])], v[:, :, :len(large_input_ids[0])]) for k, v in kv_cache
                        ]
                        large_input_ids = pred_tokens[-1:].unsqueeze(0)
                    
                    max_new_tokens = self.max_new_tokens - mismatch_pos
                    if max_new_tokens <= 0:
                        max_new_tokens = 1

            elif flag=='Not_veri':
                large_input_ids = origin_large_input_ids
                mismatch_pos = "NotValidate"
                max_new_tokens = self.max_new_tokens
                mismatch = None
                early_stop_time = None
            else:
                raise Exception
            
            if flag == "verified" or flag == "trigger":
                valid_small_output = self.tokenizer.decode(small_output_tokens[:mismatch_pos], skip_special_tokens=True)
                
                if self.task == "repoeval_line" or self.task == "repoeval_line_prefix":
                    stopping_criteria = StoppingCriteriaList([PythonLineStoppingCriteria(self.tokenizer, 0, self.NL_list)])
                else:
                    prompt = kwargs["prompt"] + valid_small_output
                    stopping_criteria = StoppingCriteriaList([PythonStatementStoppingCriteria(self.tokenizer, prompt, 0, self.parser, self.NL_list)])
                
                if large_input_ids.item() == self.tokenizer.eos_token_id or stopping_criteria(large_input_ids, None):
                    new_text = self.tokenizer.decode(large_input_ids[0], skip_special_tokens=True)
                else:
                    new_text = greedy_generate(self.model, self.tokenizer, large_input_ids, max_new_tokens, kv_cache, stopping_criteria)
                final_output = valid_small_output + new_text
            else:
                if self.task == "repoeval_line" or self.task == "repoeval_line_prefix":
                    stopping_criteria = StoppingCriteriaList([PythonLineStoppingCriteria(self.tokenizer, large_input_ids.shape[1], self.NL_list)])
                else:
                    stopping_criteria = StoppingCriteriaList([PythonStatementStoppingCriteria(self.tokenizer, kwargs["prompt"], large_input_ids.shape[1], self.parser, self.NL_list)])
                
                attention_mask = torch.ones_like(large_input_ids).to(self.device)
                outputs = self.model.generate(
                    large_input_ids,
                    attention_mask=attention_mask,  # Add attention mask
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=2 if self.model_type == "opc" else self.tokenizer.eos_token_id,
                    do_sample=False,
                    tokenizer=self.tokenizer,
                    stopping_criteria=stopping_criteria
                )
                generated_tokens = outputs[0]
                input_tokens = large_input_ids[0]

                # 找到生成文本与输入文本的起始位置
                new_tokens = generated_tokens[len(input_tokens):]

                # 使用 tokenizer 只对新的 token 进行解码
                new_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                final_output = new_text

        return final_output, mismatch, early_stop_time
