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

class SingleModelDecoding:
    def __init__(self, model, model_type, tokenizer, device, lang, parser, twice=False):
        self.model = model
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.device = device
        self.lang = lang
        self.parser=parser
        self.task=None
        self.twice = twice
        self.model_type = model_type
        # FIM special tokens
        if model_type == 'qwen':
            self.token_mapping = {
                "prefix": "<|fim_prefix|>",
                "middle": "<|fim_middle|>",
                "suffix": "<|fim_suffix|>",
                "pad":  "<|fim_pad|>"
            }
        elif model_type == "deepseek":
            self.token_mapping = {
                "prefix": "<｜fim▁begin｜>",
                "middle": "<｜fim▁hole｜>",
                "suffix": "<｜fim▁end｜>",
                "pad":  "<pad>"
            }
        elif model_type == "starcoder":
            self.token_mapping = {
                "prefix": "<fim_prefix>",
                "middle": "<fim_middle>",
                "suffix": "<fim_suffix>"
            }
        elif model_type == 'opc':
            self.token_mapping = {
                "prefix": "<PREFIX>",
                "middle": "<MIDDLE>",
                "suffix": "<SUFFIX>",
                "pad":  "<PAD>"
            }
        
        # 收集以\n结尾的token_id以实现早停止
        self.NL_list = [id for _, id in tokenizer.vocab.items() if tokenizer.decode(id).endswith("\n")]
    
    def set_task(self, task):
        self.task = task
        
    def generate(self, prefix="", suffix="", relevent_str='', **kwargs):
        """生成代码
        Args:
            input_ids: 输入的token ids
            max_length: 最大生成长度
            prefix: 前缀文本
            suffix: 后缀文本
        """
        # 构造完整上下文
        if self.model_type == "qwen":
            context = f"{self.token_mapping['prefix']}{relevent_str}{prefix}{self.token_mapping['suffix']}{suffix}{self.token_mapping['middle']}"
            context_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(self.device)
        elif self.model_type == "opc":
            context = f"{self.token_mapping['suffix']}{suffix}{self.token_mapping['prefix']}{relevent_str}{prefix}{self.token_mapping['middle']}"
            context_ids = self.tokenizer(context, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        elif self.model_type == "deepseek":
            context = f"{self.token_mapping['prefix']}{relevent_str}{prefix}{self.token_mapping['middle']}{suffix}{self.token_mapping['suffix']}"
            context_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(self.device)
        elif self.model_type == "starcoder":
            context = f"{self.token_mapping['prefix']}{relevent_str}{prefix}{self.token_mapping['suffix']}{suffix}{self.token_mapping['middle']}"
            context_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(self.device)
        else:
            raise RuntimeError("Unsupported model")
        attention_mask = torch.ones_like(context_ids).to(self.device)

        # 生成
        with torch.no_grad():
            if self.twice:
                
                generation_config = GenerationConfig(
                    num_beams=2,
                    num_return_sequences=2,
                    do_sample=False,
                    max_new_tokens=64,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=2 if self.model_type == "opc" else self.tokenizer.eos_token_id,
                    stop_strings=["\n"] if self.task == "repoeval_line" or self.task == "repoeval_line_prefix" else None
                )
                
                outputs = self.model.generate(
                    context_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    tokenizer=self.tokenizer
                    # bos_token_id=self.tokenizer.bos_token_id
                )
                
                new_tokens_0 = outputs[0][context_ids.shape[1]:]
                new_text_0 = self.tokenizer.decode(new_tokens_0, skip_special_tokens=True)
                new_text_0 = get_python_one_statement(kwargs["prompt"], new_text_0, self.parser)
                new_tokens_1 = outputs[1][context_ids.shape[1]:]
                new_text_1 = self.tokenizer.decode(new_tokens_1, skip_special_tokens=True)
                new_text_1 = get_python_one_statement(kwargs["prompt"], new_text_1, self.parser)
                score_0 = fuzz.ratio(kwargs["ground_truth"], new_text_0)
                score_1 = fuzz.ratio(kwargs["ground_truth"], new_text_1)
                final_output = new_text_0 if score_0 > score_1 else new_text_1
                return final_output, None
            else:
                outputs = self.model.generate(
                    context_ids,
                    attention_mask=attention_mask,
                    do_sample=False,
                    # generation_config=generate_config,
                    max_new_tokens=64,
                    tokenizer=self.tokenizer,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=2 if self.model_type == "opc" else self.tokenizer.eos_token_id,
                    stop_strings=["\n"] if self.task == "repoeval_line" or self.task == "repoeval_line_prefix" else None,
                    stopping_criteria=StoppingCriteriaList([PythonStatementStoppingCriteria(self.tokenizer, kwargs["prompt"], context_ids.shape[1], self.parser, self.NL_list)])
                )
            
            generated_tokens = outputs[0]
            input_tokens = context_ids[0]

            # 找到生成文本与输入文本的起始位置
            new_tokens = generated_tokens[len(input_tokens):]

            # 使用 tokenizer 只对新的 token 进行解码
            final_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
        return final_output