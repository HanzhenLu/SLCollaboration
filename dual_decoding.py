import torch
from line_profiler import profile
from typing import List
from transformers import StoppingCriteria, StoppingCriteriaList
from utils.eval_util import is_parse_valid

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

class SPDecoding:
    def __init__(self, model, model_type, tokenizer, device, lang, parser, without_sp=False):
        self.model = model
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.device = device
        self.lang = lang
        self.without_sp = without_sp
        
        self.parser = parser
        self.task=None
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
        
        if kwargs["correct"]:
            return small_output, None
        
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
                outputs = self.model(large_input_ids)
                logits = outputs.logits
                small_end_index = small_output_index + len(small_output_tokens)

                # 获取每个位置的预测token
                pred_tokens = torch.argmax(logits[0, small_output_index-1:small_end_index-1], dim=-1)
                
                mismatch = (pred_tokens != small_output_tokens).nonzero(as_tuple=True)
                if mismatch[0].numel() > 0:
                    mismatch_pos = mismatch[0][0].item()
                else:
                    mismatch_pos = len(small_output_tokens)
                
                # 保留匹配部分的小模型输出
                # Fix tensor concatenation syntax
                large_input_ids = torch.cat([origin_large_input_ids, small_output_tokens[:mismatch_pos].unsqueeze(0)], dim=1)
                max_new_tokens = 64 - mismatch_pos

            elif flag=='Not_veri':
                large_input_ids = origin_large_input_ids
                mismatch_pos = "NotValidate"
                max_new_tokens = 64
            
            elif flag=='trigger':
                mismatch_pos = "Trigger"
                max_new_tokens = 64
            else:
                raise Exception
            
            # if len(input_ids.shape) == 3:
            #     input_ids = input_ids.squeeze(0)  # Add batch dimension

            # Create attention mask (all 1s since we're not padding)
            attention_mask = torch.ones_like(large_input_ids).to(self.device)
            
            outputs = self.model.generate(
                large_input_ids,
                attention_mask=attention_mask,  # Add attention mask
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
                tokenizer=self.tokenizer,
                stop_strings=["\n"] if self.task == "repoeval_line" or self.task == "repoeval_line_prefix" else None,
                stopping_criteria=StoppingCriteriaList([PythonStatementStoppingCriteria(self.tokenizer, kwargs["prompt"], large_input_ids.shape[1], self.parser, self.NL_list)])
            )
            generated_tokens = outputs[0]
            input_tokens = large_input_ids[0]

            # 找到生成文本与输入文本的起始位置
            new_tokens = generated_tokens[len(input_tokens):]

            # 使用 tokenizer 只对新的 token 进行解码
            new_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # torch.cuda.empty_cache()
        # print("large_output:",large_output)
        # 组合最终结果
        if mismatch_pos in ['NotValidate', 'Trigger']:
            final_output =  new_text
        else:
            final_output = self.tokenizer.decode(small_output_tokens[:mismatch_pos]) + new_text
        

        return final_output
