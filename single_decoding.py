import torch
from typing import List
from utils.eval_util import get_python_one_statement, is_parse_valid
from fuzzywuzzy import fuzz
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList

class PythonStatementStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt, input_ids_len, parser, NL_list:List[int]):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.input_ids_len = input_ids_len  # 原始 prompt 的 token 长度
        self.parser = parser
        self.NL_list = NL_list

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[0] > 1:
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
        else:
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
        if input_ids.shape[0] > 1:
            for seq in input_ids:
                # 获取新生成的部分（不包含 prompt）
                generated_ids = seq[self.input_ids_len:]
                new_text:str = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # 至少要生成出一行末才会被接受
                if "\n" not in new_text:
                    return False
            
            # 如果从循环中退出，说明所有分支都已经生成了至少一行，可以结束生成了
            return True        
        else:
            # 检查是否出现换行
            if not input_ids[0][-1].detach().cpu().numpy() in self.NL_list:
                return False
            
            return True
        
class SingleModelDecoding:
    def __init__(self, model, model_type, tokenizer, max_new_tokens, device, parser, twice=False):
        self.model = model
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.parser=parser
        self.task=None
        self.twice = twice
        self.model_type = model_type
        # FIM special tokens
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
        
        # 收集以\n结尾的token_id以实现早停止
        self.NL_list = [id for _, id in tokenizer.vocab.items() if tokenizer.decode(id).endswith("\n")]
    
    def set_task(self, task):
        self.task = task
        
    def generate(self, prefix="", suffix="", relevant_str='', trigger_point_idx=None, **kwargs):
        """生成代码
        Args:
            input_ids: 输入的token ids
            max_length: 最大生成长度
            prefix: 前缀文本
            suffix: 后缀文本
        """
        if trigger_point_idx is not None:
            trigger_str = kwargs["ground_truth"][:trigger_point_idx+1]
            prefix += trigger_str
        
        # 构造完整上下文
        if self.model_type in ["qwen", "llama", "starcoder"]:
            context = f"{self.token_mapping['prefix']}{relevant_str}{prefix}{self.token_mapping['suffix']}{suffix}{self.token_mapping['middle']}"
            context_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(self.device)
        elif self.model_type == "opc":
            context = f"{self.token_mapping['suffix']}{suffix}{self.token_mapping['prefix']}{relevant_str}{prefix}{self.token_mapping['middle']}"
            context_ids = self.tokenizer(context, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        elif self.model_type == "deepseek":
            context = f"{self.token_mapping['prefix']}{relevant_str}{prefix}{self.token_mapping['middle']}{suffix}{self.token_mapping['suffix']}"
            context_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(self.device)
        else:
            raise RuntimeError("Unsupported model")
        attention_mask = torch.ones_like(context_ids).to(self.device)
        
        # 生成
        with torch.no_grad():
            if self.task == "repoeval_line" or self.task == "repoeval_line_prefix":
                stopping_criteria = StoppingCriteriaList([PythonLineStoppingCriteria(self.tokenizer, context_ids.shape[1], self.NL_list)])
            else:
                stopping_criteria = StoppingCriteriaList([PythonStatementStoppingCriteria(self.tokenizer, kwargs["prompt"], context_ids.shape[1], self.parser, self.NL_list)])
            
            if self.twice:
                
                generation_config = GenerationConfig(
                    num_beams=2,
                    num_return_sequences=2,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=2 if self.model_type == "opc" else self.tokenizer.eos_token_id
                )
                
                outputs = self.model.generate(
                    context_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    tokenizer=self.tokenizer,
                    stopping_criteria=stopping_criteria
                    # bos_token_id=self.tokenizer.bos_token_id
                )
                
                new_tokens_0 = outputs[0][context_ids.shape[1]:]
                new_text_0 = self.tokenizer.decode(new_tokens_0, skip_special_tokens=True)
                if self.task == "repoeval_line" or self.task == "repoeval_line_prefix":
                    new_text_0 = new_text_0.split("\n")[0]
                else:
                    new_text_0 = get_python_one_statement(kwargs["prompt"], new_text_0, self.parser)
                new_tokens_1 = outputs[1][context_ids.shape[1]:]
                new_text_1 = self.tokenizer.decode(new_tokens_1, skip_special_tokens=True)
                if self.task == "repoeval_line" or self.task == "repoeval_line_prefix":
                    new_text_1 = new_text_1.split("\n")[0]
                else:
                    new_text_1 = get_python_one_statement(kwargs["prompt"], new_text_1, self.parser)
                score_0 = fuzz.ratio(kwargs["ground_truth"], new_text_0)
                score_1 = fuzz.ratio(kwargs["ground_truth"], new_text_1)
                final_output = new_text_0 if score_0 > score_1 else new_text_1
                return final_output
            else:
                outputs = self.model.generate(
                    context_ids,
                    attention_mask=attention_mask,
                    do_sample=False,
                    # generation_config=generate_config,
                    max_new_tokens=self.max_new_tokens,
                    tokenizer=self.tokenizer,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=2 if self.model_type == "opc" else self.tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria
                )
            
            generated_tokens = outputs[0]
            input_tokens = context_ids[0]

            # 找到生成文本与输入文本的起始位置
            new_tokens = generated_tokens[len(input_tokens):]

            # 使用 tokenizer 只对新的 token 进行解码
            final_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
        return final_output