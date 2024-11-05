import vision_processes.vision_models as vision_models
import inspect
from inspect import signature
import torch
from dataclasses import dataclass,field
from functools import partial
from configs import config
from copy import deepcopy
import json
from PIL import Image
import torch
from torchvision import transforms
from LLM.chatgpt_function_model import PureChatGpt,LevelReviewFunction,TemplateChatGpt,FinalReviewFunction
from prompts.review_prompt import fn2ReviewLLm,Review_Prompt_for_find,Review_Prompt_for_find_V2,Review_Prompt_for_find_V3,fix_prompt_dict
import numpy as np
from vision_processes.image_patch import ImagePatch
from vision_processes.vision_models import BaseModel
from vision_processes.vision_processes import forward
import math
import openai
import traceback
import re
import json
import os
import inspect
import importlib.util
from transformers.modeling_outputs import ModelOutput
from utils.inject_utils import *
from utils.utils import number_to_ordinal
import logging
from utils.utils import get_modules_functions_from_file,MyLogger
from func_timeout import func_set_timeout
from prompts.proxy_user_prompt import proxy_user_prompt_V1,feedback_prompt_V2,analyse_prompt_V1,answer_select_prompt
from prompts.iterate_refine_prompts import intermediate_analyse_prompt,summry_prompt
# @dataclass
# class ReviewerAgent():
#     review_llm: PureChatGpt
#     reamian_times:int = 3
@dataclass
class VisionSeeker():
    queues:object
    model_name:str
    caption_cache:dict = field(default_factory=dict)

    def get_caption(self,raw_image,prompt = None,tag = None):

        if tag is not None and tag in self.caption_cache.keys():
            return  self.caption_cache[tag]
        
        if prompt is None:
            prompt = 'Describe the image in detail.'
            
        if isinstance(raw_image,ImagePatch):
            image = raw_image.cropped_image

        if self.model_name == 'blip':
            caption = forward(image = image,question = prompt,model_name=self.model_name,task='caption',queues=self.queues)
        elif self.model_name == 'IBLIP':
            caption = forward(model_name=self.model_name,image=image,prompt=prompt,queues=self.queues)

        if tag is not None:
            self.caption_cache[tag] = caption

        if isinstance(raw_image,ImagePatch):
            raw_image.caption = caption
            
        return caption
    
    def get_answer(self,image,prompt,long_answer:bool = False):
        if not long_answer:
            prompt = f'{prompt}.short answer:'
        else:
            prompt = f'{prompt}.Answer in detail:'
        answer = forward(model_name=self.model_name,image=image,prompt=prompt,queues=self.queues)
        return answer

@dataclass
class ProgWrapper():
    queues:object = None
    config: dict = config
    function_check_list:list[str] = field(default_factory=list)
    code_check: bool = False
    save_var: bool = True
    save_path:str = 'results/prog_env_result'
    log_dir:str = None
    process_id:str = '1'
    inject_html: bool = True
    replan: bool = False
    iterate_refine: bool = False
    select_answer_or_not:bool = True
    _iterate_refine_remain_time = 3 
    _replan_remain_times = 1
    simple_query_max_check_count:int = 3
    valid_query_max_check_count:int = 2
    ori_code_mode:bool = False
    select_answer_time:int = 0
    visual_grounding_task: bool =False
    def __post_init__(self):
        # with open('prompts/chatapi.prompt','r') as f:
        #     self.prompt = f.read().strip()

        self.current_state = {}
        self.all_ori_code = []
        self.all_inject_code = []
        self.all_code_snippets = []
        self.all_varpool= []        
        self.all_review_result = {}
        self.all_cs_map = []
        self.all_suggestions = []
        self.simple_query_check_count = 0
        self.find_check_count = 0
        self.valid_query_check_count = 0
        self.max_code_generate_time = 12
        self.simple_qeury_review_result = {}
        if self.log_dir is None:
            self.log_dir = os.path.join(self.save_path,'logs')
        self.logger = MyLogger(log_dir=self.log_dir,process_id=self.process_id)
        self.feedback_information = []
        self.all_answer = [None]
        self.review_dict = {}
        self.fix_prompt_dict = {}
        self.all_iterate_suggestion = []
        self.all_replan_suggestion = []
        for fn in self.function_check_list:
            d = fn2ReviewLLm.get(config.dataset_config.name)
            fd = fix_prompt_dict.get(config.dataset_config.name)
            llm = d.get(fn)
            prompt = fd.get(fn)
            self.review_dict[fn] = llm
            self.fix_prompt_dict[fn] = prompt
            self.all_review_result[fn] = {}

        self.initiative_code_generation_setting()
        self.initiative_code_execution_setting()
        
        if not config.load_models[self.config.vision_seeker_name]:
            print('Not initiative vision seeker.Default to blip')
            seeker_name = 'blip'
        else:
            seeker_name = self.config.vision_seeker_name
        self.select_answer_llm = TemplateChatGpt(prompt_template=answer_select_prompt)
        self.vision_seeker = VisionSeeker(queues=self.queues,model_name=seeker_name)
        self.final_check_llm = FinalReviewFunction(prompt_template=proxy_user_prompt_V1,parse_args={'temperature':0.1,'n':5})
        self.replan_analyser = TemplateChatGpt(prompt_template=analyse_prompt_V1)
        self.iterate_refine_analyser = TemplateChatGpt(prompt_template=intermediate_analyse_prompt)
        self.html_str = ''
        self.iterate_refine_remain_time = self._iterate_refine_remain_time
    def initiative_code_generation_setting(self):
        local_config = self.config.code_generation
        self.codellm = PureChatGpt(model=local_config.model,
                                   parse_args=dict(local_config.parse_args),
                                   logger=self.logger)
        self.input_image_varname = local_config.input_image_varname
        self.result_varname_proxy = local_config.result_varname_proxy
        self.output_type = local_config.output_type
        self.image_preprocess_function = local_config.image_preprocess
        self.generated_function_name = local_config.function_name
        with open(config.prompt_file,'r') as f:
            self.prompt = f.read().strip()

    def initiative_code_execution_setting(self):
        local_config = self.config.code_execution
        api_file = local_config.api_file
        self._init_varpool = get_modules_functions_from_file(api_file)| get_modules_functions_from_file('utils/inject_utils.py')|{'self':self}
        for name,module in self._init_varpool.items():
            if callable(module):
                sig = inspect.signature(module)
                if 'queues' in sig.parameters:
                    module = partial(module,queues = self.queues)
                    self._init_varpool[name] = module

    def initiative_varpool(self,image):
        if isinstance(image,str):
            image = Image.open(image).convert('RGB')
        self.init_varpool = self._init_varpool
        if self.image_preprocess_function is not None:
            eval(self.image_preprocess_function)
        self.init_varpool[self.input_image_varname] = image
        return self.init_varpool.copy()

    def get_info(self,inputs,name):
        
        if isinstance(inputs,ImagePatch):
            size = f' and the shape (c,h,w) of {name}:{inputs.cropped_image.shape}'
            ret = self.vision_seeker.get_caption(raw_image=inputs) + size
        elif isinstance(inputs,list):
            ret = f'lens:{len(inputs)}'
        elif isinstance(inputs,tuple):
            ret = f'lens:{len(inputs)}'
        elif inputs is None:
            ret = 'Null'
        else:
            ret = str(inputs)
        return ret

    def get_all_infos(self):
        ret = ''
        for name,infos in self.feedback_information[-1].items():
            for index,info in enumerate(infos):
                if index!=0:
                    ordinal = number_to_ordinal(index)
                    ord_name = f'The {ordinal} time of {name}'
                else:
                    ord_name = name
                ret += f'{ord_name}:{info}\n'

        return ret

    def save_intermediate_var(self,inputs,name):

        feedback_information = self.feedback_information[-1]

        if name in self.all_ori_code[-1] and not name.isdigit():
            # if isinstance(inputs,bool):
            #     name = f'if {name}?'
            info = self.get_info(inputs,name)
            if name not in feedback_information:
                feedback_information[name] = [info]
            else:
                feedback_information[name].append(info)

        self.feedback_information[-1] = feedback_information

    def to_json(self,data,file_name):
        folder_path = os.path.join(self.save_path,'json')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_name += '.json'
        file_path = os.path.join(folder_path,file_name)

        with open(file_path,'w') as f:
            json.dump(data,f,indent=2)
    
    def review(self,fn,expected_result,model_output,patch_name,parameter):
        prompt = None
        review_llm = self.review_dict.get(fn)
        object_name = patch_name.replace('_patch','').replace('_patches','')
        if fn in ('simple_query') and fn in self.function_check_list:
            if self.simple_query_check_count >self.simple_query_max_check_count or self.valid_query_check_count>self.valid_query_max_check_count:
                return None
            # for okvqa,if alreday review,do not review again
            if f'{parameter}:{model_output}' in self.all_review_result[fn]:
                return None
            
            inputs = dict(
                expected_result = expected_result,
                model_output = model_output,
                object = object_name,
                query = parameter
            )
            thinking_process,review_result = review_llm.parse(inputs,keep_history=False,return_thinking_process=True)#None:do not need modyfy.  Not None:neeed to modify.

            self.simple_query_check_count += 1
            self.all_review_result[fn][f'{parameter}:{model_output}'] = thinking_process
            
            if review_result is not None:
                self.valid_query_check_count +=1
                if isinstance(review_result,list):
                    review_result = review_result[0]          
                tag = f'{self.file_name}_{patch_name}'  
                prompt = f'The answer "{model_output}" of "{parameter}" generated by {fn} is unreasonable.{review_result}.'
                suggestion = self.fix_prompt_dict.get(fn)
                prompt += suggestion
                varpool = self.current_state.get('varpool')
                patch = varpool.get(patch_name)
                prompt = prompt.replace('IMAGE_CAPTION_INSERT_HERE',self.vision_seeker.get_caption(raw_image=patch))
                prompt = prompt.replace('EXPECTED_RESULT_INSERT_HERE',expected_result)
                prompt = prompt.replace('QUERY_INSERT_HERE',parameter)

        elif fn in ('find') and fn in self.function_check_list:
            if self.find_check_count<3:
                if model_output is not None and len(model_output) == 0: 
                    varpool = self.current_state.get('varpool')
                    query = self.current_state.get('query')
                    patch = varpool.get(patch_name)
                    image = patch.cropped_image
                    codeline = f'{patch_name}.{fn}("{parameter}")'
                    tag = f'{self.file_name}_{patch_name}'
                    patch_caption = self.vision_seeker.get_caption(raw_image=image)
                    inputs = dict(
                        object = parameter,
                        caption = patch_caption,
                        query = query
                    )
                    review_result = review_llm.parse(inputs,keep_history = False)
                    
                    self.find_check_count+=1
                    if fn not in self.all_review_result:
                        self.all_review_result[fn] = [review_result]
                    else:
                        self.all_review_result[fn].append(review_result)
                    if review_result is not None:
                        from prompts.review_prompt import Review_Prompt_for_find_V3
                        prompt = Review_Prompt_for_find_V3.replace('CODELINE',codeline).replace('OBJECT',parameter).replace('LOCATION',review_result)
                        print(f'prompt:{prompt}')

        return prompt
        
    def split_codeline_and_indent_level(self,codeline):
        origlen = len(codeline)
        codeline = codeline.lstrip()
        indent = origlen - len(codeline)
        if codeline.startswith("if"):
            code_type = 'if'
        elif codeline.startswith("elif"):
            code_type = 'elif'
        elif codeline.startswith("else"):
            code_type = 'else'
        elif codeline.startswith("for"):
            code_type = 'for'
        elif codeline.startswith("return"):
            code_type = 'assign'
        else:
            code_type = 'assign'
        return codeline,indent,code_type
    
    def get_var_name_codetype(self,codeline):
        # can output either a list of things to show, or a single thing to show
        
        things_to_show = []
        if codeline.startswith("if"):
            code_type = "if"
            try:
                condition, rest = codeline[3:].split(":", 1)
            except Exception as e:
                thing_to_show = None
                return thing_to_show,code_type
            codeline = f"if {condition}:{rest}"

            operators = ['==', '!=', '>=', '<=', '>', '<']
            things_to_show = []
            for op in operators:
                if op in condition:
                    things_to_show = [x.strip() for x in condition.split(op)]
                    # print(things_to_show)
                    break
            # things_to_show.append(thing_to_show)
            thing_to_show = things_to_show + [condition.strip()]

        elif codeline.startswith("for "):
            code_type = 'for'
            thing_to_show = codeline.split("for ")[1].split(" in ")[0]

        elif codeline.startswith("return"):
            thing_to_show = codeline.split("return ")[1]
            code_type = 'return'

        elif ' = ' in codeline:
            code_type = 'assign'
            thing_to_show = codeline.split(' = ')[0]
        elif ' += ' in codeline:
            code_type = 'assign'
            thing_to_show = codeline.split(' += ')[0]
        elif ' -= ' in codeline:
            code_type = 'assign'
            thing_to_show = codeline.split(' -= ')[0]
        elif ' *= ' in codeline:
            code_type = 'assign'
            thing_to_show = codeline.split(' *= ')[0]
        elif ' /= ' in codeline:
            code_type = 'assign'
            thing_to_show = codeline.split(' /= ')[0]

        elif '.append(' in codeline:
            code_type = 'append'
            thing_to_show = codeline.split('.append(')[0]

        elif '.extend' in codeline:
            code_type = 'extend'
            thing_to_show = codeline.split('.extend(')[0]

        elif '.add(' in codeline:
            code_type = 'add'
            thing_to_show = codeline.split('.add(')[0]

        elif '.sort(' in codeline:
            code_type = 'sort'
            thing_to_show = codeline.split('.sort(')[0]

        elif codeline.startswith("elif") or codeline.startswith("else"):
            thing_to_show = None
            code_type = 'elif_else'
        else:
            thing_to_show = None
            code_type = 'other'

        if isinstance(thing_to_show, list):
            thing_to_show = [thing if not (thing.strip().startswith("'") and thing.strip().endswith("'"))
                            else thing.replace("'", '"') for thing in thing_to_show if thing is not None]
        elif isinstance(thing_to_show, str):
            thing_to_show = thing_to_show if not (thing_to_show.strip().startswith("'") and
                                                thing_to_show.strip().endswith("'")) else thing_to_show.replace("'", '"')
        return thing_to_show, code_type
    
    def start(self,file_name,query,label):
        # self.query = query
        # if isinstance(image,str):
        #     self.image = Image.open(image).convert('RGB')
        self.html_str = ''
        self.replan_remain_times = self._replan_remain_times
        self.all_answer =[None]
        self.codellm.conversation_history=[]
        self.all_ori_code = []
        self.all_inject_code = []
        self.all_code_snippets = []
        self.all_varpool= [] 
        self.all_cs_map=[]
        self.simple_query_check_count = 0
        self.valid_query_check_count = 0
        self.find_check_count = 0
        self.file_name = file_name
        self.logger.change_sample(sample_id=file_name)
        self.current_state = {'query':query,'label':label}
        self.feedback_information = []
        self.iterate_refine_remain_time = self._iterate_refine_remain_time
        self.final_check_llm.restart()
        self.iterate_refine_analyser.restart()
        self.select_answer_llm.restart()
        self.all_iterate_suggestion = []
        self.all_replan_suggestion = []
        self.all_suggestions = []
        self.all_review_result = {}
        self.select_answer_time = 0
        for fn in self.function_check_list:
            self.all_review_result[fn] = {}
        for name,llm in self.review_dict.items():
            llm.restart()

    def inject_code(self,code):

        if self.save_var:
            code = inject_any_function(code=code,
                                         fn='self.save_intermediate_var')
        
        # if len(self.function_check_list)>0:
        #     code = self.inject_function_check(code)

        if self.inject_html:
            folder_path = os.path.join(self.save_path,'html')
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            code = inject_html_for_whole_function(code)

        return code

    def select_answer(self,query,answers):
        self.select_answer_time+=1
        inputs = dict(
            query = query,
            candidate_answers = answers
        )
        resp = self.select_answer_llm.parse(inputs,keep_history=False)[0]
        resp = resp.lower()
        try:
            final_answer = resp.split('final answer:')[-1]
        except Exception as e:
            final_answer = resp

        return final_answer
    
    def to_html(self,query,label,filename):
        html_save_dir = os.path.join(self.save_path,'html')
        if not os.path.exists(html_save_dir):
            os.makedirs(html_save_dir)
        if filename is not None:
            html_file = os.path.join(html_save_dir,f'{filename}.html')
        else:
            html_file = os.path.join(html_save_dir,f'test.html')
        html_str = self.html_str
        last_code = self.all_ori_code[-1]
        html_save(html_str,code=last_code,query=query,answer=label,html_file=html_file)

    def get_sni_from_text(self,text,stop=None):
        if stop is not None and stop in text:
            return text,text
        try:
            pattern=r"```(.+?)```"
            match=re.search(pattern,text,re.DOTALL)
            if match:
                code = match.group(0)
                code = code.replace('`','')
                code = code.replace('python\n','')
                code = code.replace('python','')
            else:
                code = text
                code_list = code.split('\n')
                ret=[]
                for line in code_list:
                    new_line = line.lstrip()
                    indent = len(line) - len(new_line)
                    if indent== 0 and not line.startswith(self.generated_function_name):
                        continue
                    else:
                        ret.append(line)
                code = '\n'.join(ret)


            ori_code = code

        except Exception as e:
            detailed_e = traceback.format_exc()
            self.logger.log(logging.INFO,detailed_e)

        if self.generated_function_name not in code and self.code_check:
            prompt = f'Generate function startswith "{self.generated_function_name}"'
            if stop is not None:
                prompt+=f'or output {stop}.'
            self.logger.log(logging.INFO,f'Did not generate function')
            return self.generate_code(prompt=prompt,stop=stop)
    

        inject_code = self.inject_code(code)
        self.all_inject_code.append(inject_code)
        return ori_code,inject_code
    
    def inject_review_codeline(self,fn,codeline):

        code,indent,type=self.split_codeline_and_indent_level(codeline)

        if type != 'assign':
            return codeline
        varname,_ = self.get_var_name_codetype(code)
        expected_result = varname
        try:
            pattern = fr'([a-zA-Z0-9_\[\]]+)\.{fn}\(([a-zA-Z0-9_\[\]\s\.,;:\'\"\!\?\-]+)\)'
            match = re.search(pattern, code)
            object_name = match.group(1)
            query = match.group(2)

        except Exception as e:
            #usually in this situation,the code is wrong
            return codeline
        
        if varname == self.result_varname_proxy:
            varname_proxy = "RP_ANS"
            expected_result = 'answer'
            #avoid return answer without review
            codeline = codeline.replace(varname,varname_proxy)
            codeline += '\n' + indent*' ' + f'REVIEW_RESULT = self.review("{fn}","{expected_result}",{varname_proxy},"{object_name}",{query})'
            codeline += '\n' + indent*' ' + f'{varname} = {varname_proxy}'
        else:
            codeline += '\n' + indent*' ' + f'REVIEW_RESULT = self.review("{fn}","{expected_result}",{varname},"{object_name}",{query})'
        
        return codeline

    def inject_function_check(self,code,check_set=None):
        check_set = self.function_check_list
        code_list = code.split('\n')
        new_code_list = []

        assert check_set is not None

        for i,code in enumerate(code_list):
            if len(code)==0:
                continue
            for fn in check_set:
                if (fn in code) and ('html' not in code):
                    code = self.inject_review_codeline(fn,code)

            new_code_list.append(code)

        new_code = '\n'.join(new_code_list)
        return new_code

    def run_program(self,code,varpool):
        self.feedback_information.append({})
        self.current_state['varpool'] = varpool
        try:
            exec(compile(code,'code','exec'),varpool)
            exec(compile('answer = execute_command(image)','code','exec'),varpool)
            answer = varpool.get('answer')
        except Exception as e:
            if self.code_check:
                prompt = f'Error in execution of statement "{self.all_ori_code[-1]}":{e}.Fix the error by generating new function startswith "def execute_command" and do not use undefined api.'
                detailed_e = traceback.format_exc()
                self.logger.log(logging.INFO,str(detailed_e))
                if len(self.all_ori_code)<self.max_code_generate_time:
                    _,new_code = self.generate_code(prompt = prompt)
                    varpool = self.init_varpool.copy()
                    return self.run_program(new_code,varpool)
                else:
                    return self.all_answer[-1]
            else:
                return e
        self.all_varpool.append(varpool)
        self.all_answer.append(answer)

        return answer
    
    def generate_code(self,query=None,prompt=None,stop=None):
        # if query is None:
        #     query=self.query
        if prompt is None and query is not None:
            prompt = self.prompt.replace('INSERT_QUERY_HERE',query)

        step = 0
        temperature = min((len(self.all_ori_code))*step,1.5)
        args = dict(
            temperature = temperature
        )
        code = self.codellm.parse(content=prompt,add_to_history= True,**args)['content']
        if stop is not None:
            if stop in code:
                return code,code
        self.logger.log(logging.INFO,code)
        ori_code = code
        ori_code,inject_code = self.get_sni_from_text(code,stop=stop)
        self.all_ori_code.append(ori_code)
        
        return ori_code,inject_code

    @func_set_timeout(900)
    def forward(self,
                query,
                image_path,
                code=None,
                file_name=None,
                label=None,
                to_json=False):
        
        self.start(file_name,query,label)
        varpool = self.initiative_varpool(image_path)

        #No code provided
        if code is None:
            ori_code,inject_code= self.generate_code(query=query)

        else:
            ori_code,inject_code = self.get_sni_from_text(code)
            self.all_ori_code.append(ori_code)
        
        try:
            _res = self.run_program(inject_code,varpool.copy())

        except Exception as e:
            return dict(
            res = str(e),
            code = inject_code
        )

        if not self.replan:
            res = _res
        else:
            res = self.replan_runprogram(_res,query,varpool.copy())

        if isinstance(res,list):
            if self.select_answer_or_not:
                res = self.select_answer(query,res)

        if self.visual_grounding_task:
            if not isinstance(res,ImagePatch):
                res = None

        if to_json:
            assert file_name is not None
            history = self.codellm.conversation_history
            if not isinstance(res,str):
                res = str(res)
            data = dict(
                    query = query,
                    image_path = image_path,
                    label = label,
                    answer = res,
                    code_generate_time = len(self.all_ori_code),
                    code = self.all_ori_code,
                    history = history[:],
                    simple_query_check_count = self.simple_query_check_count,
                    valid_query_check_count = self.valid_query_check_count,
                    find_check_count = self.find_check_count,
                    replan_remain_times = self.replan_remain_times,
                    refine_remian_time = self.iterate_refine_remain_time,
                    inject_code = self.all_inject_code,
                    all_suggestions = self.all_suggestions,
                    select_answer_time = self.select_answer_time,
                    all_feedback_info = self.feedback_information if self.save_var else None,
                    all_review_result = self.all_review_result,
                    all_answer = [str(ans) for ans in self.all_answer],
                    all_inject_code = self.all_inject_code,
                    all_replan_suggestion = self.all_replan_suggestion,
                    all_iterate_suggestion = self.all_iterate_suggestion
                    )
            self.to_json(data,file_name=str(file_name))

        if self.inject_html:
            self.to_html(query=query,
                         label=label,
                         filename=file_name)
        return dict(
            res = res,
            inject_code = inject_code
        )
    
    def replan_runprogram(self,answer,query,varpool):
        if self.replan_remain_times<=0 or len(self.all_ori_code)>self.max_code_generate_time:
            return answer
        else:
            if not self.visual_grounding_task:
                answer = str(answer)
                inputs=dict(
                    answer = answer,
                    question= query
                )
                reason,accept_or_not = self.final_check_llm.review(inputs)
            else:
                prompt = f'Can you see {query} in the image?'
                resp = self.vision_seeker.get_answer(answer.cropped_image,prompt)
                resp = resp.lower()
                if 'yes' in resp:
                    accept_or_not = True
                else:
                    accept_or_not = False
                    ori_image = self.init_varpool[self.input_image_varname]
                    question = f'Where is {query} located in the picture? Answer with upper left, lower left, upper right, or lower right.'
                    location = self.vision_seeker.get_answer(ori_image,question)
            if accept_or_not:
                if self.iterate_refine and not self.visual_grounding_task:
                    answer = self.iterate_refine_runprogram(varpool.copy(),answer)
                    inputs=dict(
                        answer = answer,
                        question= query
                    )
                    reason,accept_or_not = self.final_check_llm.review(inputs)
                    if accept_or_not:
                        return answer
                    else:
                        self.replan_remain_times -= 1
                        all_infos = self.get_all_infos()
                        inputs=dict(
                            query = query,
                            last_code = self.all_ori_code[-1],
                            reason = reason,
                            last_answer = answer,
                            interRes = all_infos
                        )
                    
                        resp = self.replan_analyser.parse(inputs,keep_history=False)
                        try:
                            suggestion = resp[0].split('Analyse:')[-1]
                        except Exception as e:
                            suggestion = resp[0]
                        prompt = f'The answer "{answer}" is unreasonable.The reason is {reason}.{suggestion}\nGenerate new code based on above suggestions.'
                        self.all_replan_suggestion.append(suggestion)
                        _,code = self.generate_code(prompt = prompt)
                        answer = self.run_program(code,varpool.copy())
                        print(f'answer after replan:{answer}\n')
                        answer = self.replan_runprogram(answer,query,varpool)
                        
                        return answer
                else:
                    return answer
            else:
                self.replan_remain_times -= 1
                if not self.visual_grounding_task:
                    all_infos = self.get_all_infos()
                    inputs=dict(
                        query = query,
                        last_code = self.all_ori_code[-1],
                        reason = reason,
                        last_answer = answer,
                        interRes = all_infos
                    )
                
                    resp = self.replan_analyser.parse(inputs,keep_history=False)
                    try:
                        suggestion = resp[0].split('Analyse:')[-1]
                    except Exception as e:
                        suggestion = resp[0]
                    self.all_replan_suggestion.append(suggestion)
                    prompt = f'The answer "{answer}" is unreasonable.The reason is {reason}.{suggestion}\nGenerate new code based on above suggestions.'
                else:
                    prompt = f'The possible location of {query} is {location}.'
                _,code = self.generate_code(prompt = prompt)
                answer = self.run_program(code,varpool.copy())
                print(f'answer after replan:{answer}\n')
                answer = self.replan_runprogram(answer,query,varpool)

        return answer

    def iterate_refine_runprogram(self,varpool,answer):
        for i in range(self.iterate_refine_remain_time):
            inter_res= self.get_all_infos()
            print(inter_res)
            inputs = dict(
                return_value = answer,
                code = self.all_ori_code[-1],
                inter_res = inter_res
            )
            resp = self.iterate_refine_analyser.parse(inputs,keep_history=False)[0]
            self.all_iterate_suggestion.append(resp)
            query = self.current_state.get('query')
            stop = 'COMPLETE'
            fix_prompt = f'''
The above is the analysis result after executing your code. 
Imagine you are a professional Python code engineer. 
Are these intermediate variables consistent with your expected results ?
Do you think you can improve your code based on the information above to solve the visual problem {query}? 
If you think it's possible, output the new 'def execute_command' function with ```python``` and explan why.
If not, do not need to geneate function ,just output {stop} and explan why,.
'''         
            prompt = resp + fix_prompt
            ori_code,code = self.generate_code(prompt = prompt,stop=stop)
            self.iterate_refine_remain_time-=1
            if ori_code in self.all_ori_code[:-1] or stop in ori_code:
                answer = self.all_answer[-1]
                break
            answer = self.run_program(code,varpool.copy())

            print(f'answer after iteration {i}:{answer}')
        return answer
    


