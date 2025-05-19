import torch
from line_profiler import profile
from torch import nn
from transformers import GenerationConfig
from utils.eval_util import process_example_inline

class SPDecoding:
    def __init__(self, l_model, l_model_type, l_tokenizer, device, lang, s_model, s_model_type, s_tokenizer, parser, task=None, without_sp=False):
        self.l_model = l_model
        self.l_model_type = l_model_type
        self.s_model = s_model
        self.s_model_type = s_model_type
        
        self.l_tokenizer = l_tokenizer
        self.s_tokenizer = s_tokenizer
        self.device = device
        self.lang = lang
        self.without_sp = without_sp
        
        self.parser = parser
        self.task=None
        self.token_mapping = {
            "s_prefix": "<PREFIX>",
            "s_middle": "<MIDDLE>",
            "s_suffix": "<SUFFIX>",
            "l_prefix": "<|fim_prefix|>",
            "l_middle": "<|fim_middle|>",
            "l_suffix": "<|fim_suffix|>",
        }
        if 'qwen' == l_model_type:
            self.token_mapping["l_prefix"] = "<|fim_prefix|>"
            self.token_mapping["l_middle"] = "<|fim_middle|>"
            self.token_mapping["l_suffix"] = "<|fim_suffix|>"
            self.token_mapping["l_pad"] = "<|fim_pad|>"
        elif 'deepseek' == l_model_type:
            self.token_mapping["l_prefix"] = '<｜fim▁begin｜>'
            self.token_mapping["l_middle"] = '<｜fim▁hole｜>'
            self.token_mapping["l_suffix"] = '<｜fim▁end｜>'
        elif "starcoder" == l_model_type:
            self.token_mapping["l_prefix"] = "<fim_prefix>"
            self.token_mapping["l_middle"] = "<fim_middle>"
            self.token_mapping["l_suffix"] = "<fim_suffix>"
        
        # 定义需要大模型介入的特殊关键字
        self.special_keywords = {
            'python': ['if', 'while', 'for', 'def', 'class', 'try', 'with'],
            'java': ['if', 'while', 'for', 'switch', 'try', 'synchronized']
        }
    
    def set_task(self,task):
        self.task = task
    
    def need_large_model(self, code: str) -> bool:
        """判断是否需要大模型介入"""
        # 检查是否以换行符结束
        if code.endswith('\n'):
            return True
            
        # 检查是否包含特殊关键字
        # for keyword in self.special_keywords.get(self.lang, []):
        #     if keyword in code.split():
        #         return True
                
        return True
    
    @profile
    def generate(self, input_ids, max_length=128, prefix="", suffix="",relevent_str="",key='',ground_truth="",trigger_point_idx=None, **kwargs):
        """生成代码"""
        # 使用小模型生成单行代码
        
        # print(input_ids.shape)
        # Get attention mask for input_ids if not provided
        # if isinstance(input_ids, torch.Tensor) and len(input_ids.shape) == 2:
        #     # attention_mask = torch.ones_like(input_ids)
        # else:
            # If input_ids is from tokenizer output, create proper attention mask
            # attention_mask = torch.ones((1, input_ids.shape[0]), dtype=torch.long)
        input_ids = input_ids.unsqueeze(0)  # Add batch dimension
        # print('small input tokens:',self.s_tokenizer.decode(input_ids[0]))

        small_output = self._generate_with_model(
            model=self.s_model,
            tokenizer=self.s_tokenizer,
            input_ids=input_ids,
            is_small_model=True
        )

        small_output, correct = process_example_inline("python", self.parser, small_output, ground_truth, kwargs["prompt"])
        if correct:
            return small_output, None
        
        if trigger_point_idx is not None:
            trigger_str = ground_truth[:trigger_point_idx]
            small_output = trigger_str
            # small_output = ''
        else:
            trigger_str = ""
        small_output = small_output + self.l_tokenizer.eos_token
        context = f"{self.token_mapping['l_prefix']}{relevent_str}{prefix}{self.token_mapping['l_suffix']}{suffix}{self.token_mapping['l_middle']}{small_output}"
        origin_context = f"{self.token_mapping['l_prefix']}{relevent_str}{prefix}{self.token_mapping['l_suffix']}{suffix}{self.token_mapping['l_middle']}"

        large_input_ids = self.l_tokenizer(context, return_tensors="pt").input_ids.to(self.device)
        origin_large_input_ids = self.l_tokenizer(origin_context, return_tensors="pt").input_ids.to(self.device)

        small_output_tokens = self.l_tokenizer(small_output, return_tensors="pt").input_ids.to(self.device)
        small_output_index = origin_large_input_ids.shape[1]

        if self.without_sp:
            flag = 'Not_veri'
        else:
            flag = 'verified'
        
        if trigger_point_idx is not None:
            flag = 'trigger'
        
        large_output, mismatch_pos = self.generate_with_large_model(
            model=self.l_model,
            tokenizer=self.l_tokenizer,
            input_ids=large_input_ids,
            max_length=max_length,
            small_tokens=small_output_tokens[0],
            small_output_index=small_output_index,
            flag = flag,
            origin_context = origin_large_input_ids
        )

        # torch.cuda.empty_cache()
        # print("large_output:",large_output)
        # 组合最终结果
        if mismatch_pos == 'ReDecoding' or mismatch_pos=='NotValidate' or mismatch_pos=="Trigger":
            final_output =  large_output 
        elif mismatch_pos == 'Small':
            final_output = small_output
        else:
            final_output = self.l_tokenizer.decode(small_output_tokens[0,:mismatch_pos]) + large_output 
        

        return final_output, small_output

    
    def _generate_with_model(self, model, tokenizer, input_ids, max_new_tokens=64, is_small_model=True, generation_config=None):
        """使用指定模型生成代码"""
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            
            if len(input_ids.shape) == 3:
                input_ids = input_ids.squeeze(0)  # Add batch dimension

            # print(input_ids.shape)
            # Create attention mask (all 1s since we're not padding)
            attention_mask = torch.ones_like(input_ids).to(self.device)

            # print(attention_mask)
            
            if generation_config!=None:
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,  # Add attention mask
                    # max_length=input_ids.size(1)+64,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=2 if is_small_model else tokenizer.eos_token_id,
                    do_sample=False,
                    # temperature=0,
                    # top_p=0.95,
                    generation_config=generation_config,
                    tokenizer=tokenizer
                )
                if generation_config.num_return_sequences == 2:
                    generated_tokens = outputs[1]
                else:
                    generated_tokens = outputs[0]
                    
            else:
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,  # Add attention mask
                    # max_length=max_length,
                    #新token数量
                    # max_length=input_ids.size(1)+64,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=2 if is_small_model else tokenizer.eos_token_id,
                    do_sample=False,
                    # temperature=0,
                    # top_p=0.95,
                    tokenizer=tokenizer
                )
                generated_tokens = outputs[0]
            input_tokens = input_ids[0]

            # 找到生成文本与输入文本的起始位置
            new_tokens = generated_tokens[len(input_tokens):]

            # 使用 tokenizer 只对新的 token 进行解码
            new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            # print('new text:',new_text)

            return new_text
    
    @profile
    def generate_with_large_model(self, model, tokenizer, input_ids, max_length, small_tokens, small_output_index, flag, origin_context):
        # small_output_index 是利用大模型tokenize后，小模型第一个输出的token在大模型的输出位置
        # flag 连接处是否是分开的token 还是是连在一起的token
        
        with torch.no_grad():
            if flag.lower()=='verified':
                outputs = self.l_model(input_ids)
                logits = outputs.logits
                small_end_index = small_output_index + len(small_tokens)

                # 获取每个位置的预测token
                pred_tokens = torch.argmax(logits[0, small_output_index-1:small_end_index-1], dim=-1)
                
                # print('small:',small_tokens)
                # print("pred:",pred_tokens)
                # 找到第一个不匹配的位置
                # mismatch_pos = None
                # for i, (pred, small) in enumerate(zip(pred_tokens, small_tokens)):
                #     if pred != small:
                #         mismatch_pos = i
                #         break
                mismatch = (pred_tokens != small_tokens).nonzero(as_tuple=True)
                if mismatch[0].numel() > 0:
                    mismatch_pos = mismatch[0][0].item()
                else:
                    mismatch_pos = None
                
                if mismatch_pos is not None:
                    # 保留匹配部分的小模型输出
                    # Fix tensor concatenation syntax
                    input_ids = torch.cat([origin_context, small_tokens[:mismatch_pos].unsqueeze(0)], dim=1)
                    max_new_tokens = 64 - mismatch_pos

                else:
                    # 让大模型使用beamsearch生成
                    generation_config = GenerationConfig(
                        num_beams=2,
                        num_return_sequences=2,
                        do_sample=False,
                        max_new_tokens=64,
                        pad_token_id=self.l_tokenizer.pad_token_id,
                        eos_token_id=self.l_tokenizer.eos_token_id,
                        # stop_strings=['\n']
                    )
                    
                    large_output = self._generate_with_model(
                        model=self.l_model,
                        tokenizer=self.l_tokenizer,
                        input_ids=origin_context,
                        is_small_model=False,
                        generation_config=generation_config
                    )
                    large_output = large_output.replace(self.token_mapping['l_pad'],'')
                    return large_output, "ReDecoding"
                    # small_str = tokenizer.decode(small_tokens,skip_special_tokens=True)
                    # return small_str, "Small"
            elif flag=='Not_veri':
                input_ids = origin_context
                mismatch_pos = "NotValidate"
                max_new_tokens = 64
            
            elif flag=='trigger':
                input_ids = input_ids
                mismatch_pos = "Trigger"
                max_new_tokens = 64
            else:
                raise Exception

            large_output = self._generate_with_model(
                model=self.l_model,
                tokenizer=self.l_tokenizer,
                max_new_tokens=max_new_tokens,
                input_ids=input_ids,
                is_small_model=False
            )
            large_output = large_output.replace(self.token_mapping['l_pad'],'')

            return large_output, mismatch_pos
            # return
            
