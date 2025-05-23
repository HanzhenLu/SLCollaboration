import torch
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
        for seq in input_ids:
            # 获取新生成的部分（不包含 prompt）
            generated_ids = seq[self.input_ids_len:]
            new_text:str = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # 至少要生成出一行末才会被接受
            if "\n" not in new_text:
                return False
            
            code = self.prompt
            new_text_lines = new_text.splitlines(keepends=True)
            parse_valid = False
            for line in new_text_lines:
                code = code + line
                if is_parse_valid(self.parser, code):
                    parse_valid = True
                    break
            
            # 说明这一支仍然没生成出能被正确解析的语句，需要继续生成
            if not parse_valid:
                return False
        
        # 如果从循环中退出，说明所有分支都已经有了能够被正确解析的语句，可以结束生成了
        return True
        
class PythonLineStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, input_ids_len):
        self.tokenizer = tokenizer
        self.input_ids_len = input_ids_len  # 原始 prompt 的 token 长度

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for seq in input_ids:
            # 获取新生成的部分（不包含 prompt）
            generated_ids = seq[self.input_ids_len:]
            new_text:str = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # 至少要生成出一行末才会被接受
            if "\n" not in new_text:
                return False
        
        # 如果从循环中退出，说明所有分支都已经生成了至少一行，可以结束生成了
        return True     

class SPDecoding:
    def __init__(self, model, model_type, tokenizer, device, lang, parser, without_sp=False):
        self.model = model
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.device = device
        self.lang = lang
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

        self.NL_list = [id for _, id in tokenizer.vocab.items() if tokenizer.decode(id).endswith("\n")]
        
    def set_task(self,task):
        self.task = task
    
    @profile
    def generate(self, prefix="", suffix="", relevant_str="", ground_truth="", trigger_point_idx=None, **kwargs):
        """生成代码"""
        small_output = kwargs["small_output"]
        
        # if kwargs["correct"]:
        #     return small_output, None
        
        if trigger_point_idx is not None:
            trigger_str = ground_truth[:trigger_point_idx]
            small_output = trigger_str
            flag = "trigger"
        elif self.without_sp:
            flag = "Not_veri"
        else:
            flag = "verified"
        
        if self.model_type == "qwen" or self.model_type == "starcoder":
            context = f"{self.token_mapping['prefix']}{relevant_str}{prefix}{self.token_mapping['suffix']}{suffix}{self.token_mapping['middle']}{small_output}"
            origin_context = f"{self.token_mapping['prefix']}{relevant_str}{prefix}{self.token_mapping['suffix']}{suffix}{self.token_mapping['middle']}"
        elif self.model_type == "deepseek":
            context = f"{self.token_mapping['prefix']}{relevant_str}{prefix}{self.token_mapping['middle']}{suffix}{self.token_mapping['suffix']}{small_output}"
            origin_context = f"{self.token_mapping['prefix']}{relevant_str}{prefix}{self.token_mapping['middle']}{suffix}{self.token_mapping['suffix']}"
        else:
            raise RuntimeError(f"Unsupported model_type: {self.model_type}")

        large_input_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(self.device)
        origin_large_input_ids = self.tokenizer(origin_context, return_tensors="pt").input_ids.to(self.device)

        small_output_tokens = self.tokenizer(small_output, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)[0]
        small_output_index = origin_large_input_ids.shape[1]
        
        with torch.no_grad():
            if flag == 'verified':
                if len(small_output_tokens) == 0:
                    large_input_ids = origin_large_input_ids
                    mismatch_pos = "NotValidate"
                    max_new_tokens = 64
                    flag = "Not_veri"
                else:
                    outputs = self.model(large_input_ids, use_cache=True)
                    logits = outputs.logits
                    kv_cache = outputs.past_key_values
                    small_end_index = small_output_index + len(small_output_tokens)

                    # 获取每个位置的预测token
                    pred_tokens = torch.argmax(logits[0, small_output_index-1:small_end_index], dim=-1)
                    
                    mismatch = (pred_tokens[:-1] != small_output_tokens).nonzero(as_tuple=True)
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
                    
                    max_new_tokens = 64 - mismatch_pos
                    if max_new_tokens <= 0:
                        max_new_tokens = 1

            elif flag=='Not_veri':
                large_input_ids = origin_large_input_ids
                mismatch_pos = "NotValidate"
                max_new_tokens = 64
            
            elif flag=='trigger':
                mismatch_pos = "Trigger"
                max_new_tokens = 64
            else:
                raise Exception
            
            if flag == "verified":
                valid_small_output = self.tokenizer.decode(small_output_tokens[:mismatch_pos], skip_special_tokens=True)
                
                if self.task == "repoeval_line" or self.task == "repoeval_line_prefix":
                    stopping_criteria = StoppingCriteriaList([PythonLineStoppingCriteria(self.tokenizer, 0)])
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
                    stopping_criteria = StoppingCriteriaList([PythonLineStoppingCriteria(self.tokenizer, large_input_ids.shape[1])])
                else:
                    stopping_criteria = StoppingCriteriaList([PythonStatementStoppingCriteria(self.tokenizer, kwargs["prompt"], large_input_ids.shape[1], self.parser, self.NL_list)])
                
                attention_mask = torch.ones_like(large_input_ids).to(self.device)
                outputs = self.model.generate(
                    large_input_ids,
                    attention_mask=attention_mask,  # Add attention mask
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
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
                final_output =  new_text

        return final_output
