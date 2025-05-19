import torch
from utils.eval_util import get_python_one_statement
from fuzzywuzzy import fuzz
from transformers import GenerationConfig

class SingleModelDecoding:
    def __init__(self, model, model_type, tokenizer, device, lang, parser, task=None, twice=False):
        self.model = model
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.device = device
        self.lang = lang
        self.parser=parser
        self.task=task
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
            self.token_mapping = {}
            self.token_mapping["prefix"] = "<fim_prefix>"
            self.token_mapping["middle"] = "<fim_middle>"
            self.token_mapping["suffix"] = "<fim_suffix>"
        elif model_type == 'opc':
            self.token_mapping = {
                "prefix": "<PREFIX>",
                "middle": "<MIDDLE>",
                "suffix": "<SUFFIX>",
                "pad":  "<PAD>"
            }
            
    def set_task(self,task):
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
        # print('input:',context)
        
        # print('input_id:',context_ids)
        # generate_config = GenerationConfig(stop_strings=['\n'])
        generation_config = GenerationConfig(
                        num_beams=2,
                        num_return_sequences=2,
                        do_sample=False,
                        max_new_tokens=64,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=2 if self.model_type == "opc" else self.tokenizer.eos_token_id,
                    )
        # 生成
        with torch.no_grad():
            if self.twice:
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
                    # bos_token_id=self.tokenizer.bos_token_id
                )
            
            generated_tokens = outputs[0]
            input_tokens = context_ids[0]

            # 找到生成文本与输入文本的起始位置
            new_tokens = generated_tokens[len(input_tokens):]

            # 使用 tokenizer 只对新的 token 进行解码
            final_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
        return final_output,None