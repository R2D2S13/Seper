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
import logging
from utils.utils import get_modules_functions_from_file,MyLogger
from func_timeout import func_set_timeout
from prompts.proxy_user_prompt import proxy_user_prompt_V1,feedback_prompt_V2,analyse_prompt_V1
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
        caption = forward(model_name=self.model_name,image=image,prompt=prompt,queues=self.queues)
        return caption

@dataclass
class ProgWrapper():
    queues:object = None
    config: dict = config
    function_check_list:list[str] = field(default_factory=list)
    code_check:bool = False
    save_var:bool = True
    save_path:str = 'results/prog_env_result'
    log_dir:str = None
    process_id:str = '1'
    inject_html:bool = True
    replan:bool = False
    iterate_refine:bool = True
    _replan_remain_times = 3
    _iterate_remain_times = 3
    simple_query_max_check_count:int = 3
    valid_query_max_check_count:int = 2
    ori_code_mode:bool = False

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
        self.max_code_generate_time = 9
        self.simple_qeury_review_result = {}
        if self.log_dir is None:
            self.log_dir = os.path.join(self.save_path,'logs')
        self.logger = MyLogger(log_dir=self.log_dir,process_id=self.process_id)
        self.feedback_information = {}
        self.all_answer = [None]
        self.review_dict = {}
        self.fix_prompt_dict = {}
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
        self.vision_seeker = VisionSeeker(queues=self.queues,model_name=seeker_name)
        self.final_check_llm = FinalReviewFunction(prompt_template=proxy_user_prompt_V1,parse_args={'temperature':0.1,'n':7})
        self.analyser = TemplateChatGpt(prompt_template=analyse_prompt_V1)

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
            ret = self.vision_seeker.get_caption(raw_image=inputs)
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
        for name,info in self.feedback_information.items():
            ret += f'{name}:{info}\n'
        return ret

    def save_intermediate_var(self,inputs,name):

        feedback_information = self.feedback_information

        if name in self.all_ori_code[-1] and not name.isdigit():
            if isinstance(inputs,bool):
                name = f'if {name}?'
            info = self.get_info(inputs,name)
            feedback_information[name] = info

        self.feedback_information = feedback_information

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
    
    def start(self,file_name,query):
        # self.query = query
        # if isinstance(image,str):
        #     self.image = Image.open(image).convert('RGB')
        self.replan_remain_times = self._replan_remain_times
        self.iterate_refine_times = self._iterate_remain_times
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
        self.current_state = {'query':query}
        self.feedback_information = {}
        self.final_check_llm.restart()
        self.all_suggestions = []
        self.all_review_result = {}
        for fn in self.function_check_list:
            self.all_review_result[fn] = {}
        for name,llm in self.review_dict.items():
            llm.restart()

    def inject_code(self,code):

        if self.save_var:
            code = inject_any_function_V3(code=code,
                                         fn='self.save_intermediate_var')
        
        if len(self.function_check_list)>0:
            code = self.inject_function_check(code)

        if self.inject_html:
            folder_path = os.path.join(self.save_path,'html')
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            html_file = os.path.join(folder_path,f'{self.file_name}.html')
            code = inject_html_v3(code,
                                    html_file=html_file,
                                    final_var_name=self.result_varname_proxy)

        return code
    
    def get_sni_from_text(self,text):
        if self.ori_code_mode:
            return self.get_sni_from_text_ori(self,text)
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
            def_indent = -1
            ret = []
            start = 0
            for i,line in enumerate(code_list):
                line_strip = line.strip()
                if line_strip.startswith(self.generated_function_name):
                    start = i
                    break
                elif line_strip.startswith('image_patch =') or line_strip.startswith('image_patch='):
                    start = i
                    break
            code_list = code_list[start:]

            first_code = code_list[0]
            second_code = code_list[1]
            if not first_code.startswith(self.generated_function_name):
                first_code_strip = first_code.lstrip()
                second_code_strip = second_code.lstrip()
                frist_indent = len(first_code)-len(first_code_strip)
                second_indent = len(second_code)-len(second_code_strip)
                if frist_indent != second_indent:
                    dif = second_indent - frist_indent
                    first_code = max(0,dif) * ' '+ first_code_strip
                ret.append([first_code])
                code_list = code_list[1:]
                for line in code_list:
                    if len(line) == 0:
                        continue
                    if not line.startswith(self.generated_function_name):
                        ret[-1].append(line)
                    else:
                        break
            else:
                for line in code_list:
                    if len(line)==0:
                        continue
                    new_line = line.lstrip()
                    indent = len(line) - len(new_line)
                    if indent<= def_indent:
                        def_indent = -1
                    if self.generated_function_name is not None and self.generated_function_name in text:
                        if line.startswith(self.generated_function_name):
                            ret.append([line])
                            new_line = line.lstrip()
                            def_indent = len(line) - len(new_line)

                        if not line.startswith(self.generated_function_name) and def_indent != -1:
                            if indent > def_indent:
                                ret[-1].append(line)
                    else:
                        if len(ret)==0:
                            ret.append([line])
                        else:
                            ret[-1].append(line)
            
        except Exception as e:
            detailed_e = traceback.format_exc()
            self.logger.log(logging.INFO,detailed_e)

        if len(ret)==0 and self.code_check:
            prompt = 'Generate function startswith "def execute_command"'
            self.logger.log(logging.INFO,f'Did not generate function:{text}')
            return self.generate_code(prompt=prompt)
        
        ori_code = '\n'.join(ret[0])
        print(ori_code)
        if self.output_type == 'return':
            code = ori_code.replace('return ',f'{self.result_varname_proxy} = ')

        if self.result_varname_proxy not in code and self.code_check:
            if self.output_type == 'return':
                prompt = 'There are no return statement in the code.Generate function startswith "def execute_command'
            elif self.output_type == 'varname':
                prompt = f'You did not define {self.result_varname_proxy} variable in the code'
            return self.generate_code(prompt=prompt)
        
        ori_code_snippets,inject_or_not = self.code2cs(code)
        self.all_code_snippets.append(ori_code_snippets)
        
        inject_code_snippets = []
        cs_map = {}
        new_inject_code = ''
        if self.inject_html:
            ori_code_snippets.insert(0,['html_str=""',0])
            inject_or_not.insert(0,True)
        for index,(cs,inject) in enumerate(zip(ori_code_snippets,inject_or_not)):
            if inject:
                if isinstance(cs[0],str):
                    code,indent= cs
                    inject_code = self.inject_code(code)
                    new_inject_code+=inject_code + '\n'
                    inject_code_list = inject_code.split('\n')
                    for codeline in inject_code_list:
                        inject_code_snippets.append([codeline.strip(),indent])
                        cs_map[len(inject_code_snippets)-1] = index
                        

                elif isinstance(cs[0],list):
                    new_cs = []
                    for i,codeblock in enumerate(cs):
                        code,indent= codeblock
                        inject_code = self.inject_code(code)
                        
                        inject_code_list = inject_code.split('\n')
                        for codeline in inject_code_list:
                            new_code_line,new_indent,_ = self.split_codeline_and_indent_level(codeline)
                            new_indent = indent + new_indent
                            new_cs.append([new_code_line,new_indent])
                            new_inject_code += new_indent*' '+new_code_line + '\n'

                    inject_code_snippets.append(new_cs) 
                    cs_map[len(inject_code_snippets)-1] = index
            else:
                inject_code_snippets.append(cs)
                cs_map[len(inject_code_snippets)-1] = index

        self.all_cs_map.append(cs_map)
        self.all_inject_code.append(new_inject_code)
        
        return ori_code,inject_code_snippets

    def code2cs(self,code):
        code_list = code.split('\n')
        code_snippets=[]
        snippet=[]
        inject_or_not = []
        crackect_tag = False
        if code_list[0].startswith(self.generated_function_name):
            code_list = code_list[1:]
        _,last_min_indent,t=self.split_codeline_and_indent_level(code_list[0])
        if last_min_indent == 0:
            indent_fix_value = 0
        elif last_min_indent == 4:
            indent_fix_value = 4
        else:
            indent_fix_value = last_min_indent
        for line in code_list:
            
            codeline,indent,type=self.split_codeline_and_indent_level(line)

            if len(codeline)==0 or codeline.startswith('#') or codeline.startswith("def"):
                continue

            if indent <= last_min_indent and type == 'assign':

                if len(snippet)>0 and not crackect_tag:
                    code_snippets.append(snippet)
                    inject_or_not.append(True)
                    snippet =[]

                if codeline.endswith('['):
                    crackect_tag = True
                    snippet.append([codeline,indent-indent_fix_value])

                elif codeline.endswith(']') and crackect_tag:
                    snippet.append([codeline,indent-indent_fix_value])
                    code_snippets.append(snippet)
                    inject_or_not.append(False)
                    crackect_tag = False
                    snippet =[]

                else:
                    code_snippets.append([codeline,indent-indent_fix_value])
                    inject_or_not.append(True)

            else:
                if type =='assign':
                    snippet.append([codeline,indent-indent_fix_value])
                else:
                    if indent>last_min_indent:
                        snippet.append([codeline,indent-indent_fix_value])
                    elif type=='else' or type =='elif':
                        snippet.append([codeline,indent-indent_fix_value])
                    elif len(snippet)==0:
                        snippet.append([codeline,indent-indent_fix_value])
                    else:
                        code_snippets.append(snippet)
                        inject_or_not.append(True)
                        snippet = []
                        snippet.append([codeline,indent-indent_fix_value])

            last_min_indent = min(last_min_indent,indent)

        if len(snippet)>0:
            code_snippets.append(snippet)
            inject_or_not.append(True)

        return code_snippets,inject_or_not

    def cs2code(self,cs):
        
        if isinstance(cs[0],list):
            new_code = ''
            for line in cs:
                try:
                    code,indent = line[0],line[1]
                    indent = int(indent)
                    new_code += indent*' '+ code +'\n'
                except Exception as e:
                    print(e)

        elif isinstance(cs[0],str):
            new_code = ''
            code,indent = cs[0],cs[1]
            indent = int(indent)
            new_code += indent*' '+ code +'\n'
        return new_code
    
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

    

    def run_program(self,code_snippets,varpool,start=0):
        self.feedback_information = {}
        try:
            self.current_state['varpool'] = varpool
            for i,snippet in enumerate(code_snippets):
                if i<start:
                    continue
                if self.result_varname_proxy in varpool.keys():
                    return varpool[self.result_varname_proxy]
                new_code=''
                if isinstance(snippet[0],str):
                    code,indent=snippet[0],snippet[1]
                    new_code=indent * " "+code+'\n'
                else:
                    new_code = self.cs2code(snippet)
                status,res = self.step(codeline=new_code,varpool=varpool,index = i)
                if status != 0:
                    break

        except Exception as e:
            traceback.print_exc()
            
        if self.result_varname_proxy not in varpool.keys() and status==0:
            res = 'There is no return statement in the function'
            status = 1

        self.all_varpool.append(varpool)

        if self.code_check:
        # 启用code_check,未启用review_bool：只检查代码执行对错
        # 同时启用：检查代码执行对错，并且检查simple_query执行结果
        # 都不用：常规viper
            inputs = (status,res,varpool)
            final_res = self.check(inputs)
            
        else:
            if status != 1:
                res = varpool[self.result_varname_proxy]
            final_res = res
            
        self.all_answer.append(final_res)
        return final_res
    
    def generate_code(self,query=None,prompt=None):
        # if query is None:
        #     query=self.query
        if prompt is None and query is not None:
            prompt = self.prompt.replace('INSERT_QUERY_HERE',query)

        if len(self.all_ori_code)>self.max_code_generate_time:
            self.all_ori_code.append('repeat last code')
            self.all_code_snippets.append('repeat last code')
            return self.all_ori_code[-1],self.all_code_snippets[-1]
        step = 0
        temperature = min((len(self.all_ori_code))*step,1.5)
        args = dict(
            temperature = temperature
        )
        code = self.codellm.parse(content=prompt,add_to_history= True,**args)['content']
        print(code)
        self.logger.log(logging.INFO,code)
        ori_code = code
        _,code_snippets = self.get_sni_from_text(code)

        self.all_ori_code.append(ori_code)
        self.all_code_snippets.append(code_snippets)
        
        return ori_code,code_snippets

    @func_set_timeout(900)
    def forward(self,
                query,
                image_path,
                code=None,
                file_name=None,
                label=None,
                to_json=False):
        
        self.start(file_name,query)
        varpool = self.initiative_varpool(image_path)

        #No code provided
        if code is None:
            ori_code,first_code_snippet = self.generate_code(query=query)

        else:
            ori_code,first_code_snippet = self.get_sni_from_text(code)
            self.all_ori_code.append(ori_code)
            self.all_code_snippets.append(first_code_snippet)

        if len(first_code_snippet)==0:
            return None
        
        try:
            _res = self.run_program(first_code_snippet,varpool.copy())

        except Exception as e:
            return dict(
            res = str(e),
            ori_code= ori_code,
            code_snippets= first_code_snippet
        )

        if not self.replan:
            res = _res
        else:
            res = self.replan_runprogram(_res,query,varpool.copy())

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
                    inject_code = self.all_inject_code,
                    all_suggestions = self.all_suggestions,
                    all_feedback_info = self.feedback_information if self.save_var else None,
                    all_review_result = self.all_review_result,
                    all_answer = [str(ans) for ans in self.all_answer]
                    )
            self.to_json(data,file_name=str(file_name))

        return dict(
            res = res,
            ori_code= ori_code,
            code_snippets= first_code_snippet
        )
    
    def replan_runprogram(self,answer,query,varpool):
        answer = str(answer)
        if self.replan_remain_times<=0 or len(self.all_ori_code)>self.max_code_generate_time:
            return answer
        else:
            inputs=dict(
                answer = answer,
                question= query
            )
            reason,accept_or_not = self.final_check_llm.review(inputs)
            print(f'reason:{reason}')
            if accept_or_not:
                if self.iterate_refine:
                    return self.iterate_refine_runprogram(varpool.copy())
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
            
                resp = self.analyser.parse(inputs,keep_history=False)
                try:
                    suggestion = resp[0].split('Analyse:')[-1]
                except Exception as e:
                    suggestion = resp[0]
                prompt = f'The answer "{answer}" is unreasonable.The reason is {reason}.{suggestion}\nGenerate new code based on above suggestions.'
                self.all_suggestions.append(suggestion)
                _,code_snippet = self.generate_code(prompt = prompt)
                answer = self.run_program(code_snippet,varpool.copy())
                print(f'answer after replan:{answer}\n')
                answer = self.replan_runprogram(answer,query,varpool)

        return answer



    def iterate_refine_runprogram(self,varpool):
        for i in range(self.iterate_refine_times):
            intermediate_step = self.get_all_infos()
            prompt = f'Here are the intermediate steps of your previous code.\n{intermediate_step}.\nBased on the intermediate result,Can you refine your code?'
            _,code_snippet = self.generate_code(prompt = prompt)
            answer = self.run_program(code_snippet,varpool.copy())
            print(f'answer after iteration {i}:{answer}')
        return answer
    
    def check(self,inputs):
        status, content, varpool = inputs
        # program success
        if status == 0:
            res = varpool[self.result_varname_proxy]
        else:
            if len(self.all_ori_code)>self.max_code_generate_time:
                return content
            
            #logic wrong
            elif status == 2:
                prompt = content
                resp = self.codellm.parse(prompt,add_to_history=True)['content']
                new_code,new_code_sni= self.get_sni_from_text(resp)
                self.all_ori_code.append(new_code)
                self.all_code_snippets.append(new_code_sni)
                start = 0

                # for line1,line2 in zip(new_code_sni,self.all_code_snippets[-2]):
                #     if line1 == line2:
                #         start += 1
                #     else:
                #         break

                if start==0:
                    varpool = self.init_varpool.copy()

                if self.result_varname_proxy in varpool.keys():
                    del varpool[self.result_varname_proxy]

                if 'REVIEW_RESULT' in varpool.keys():
                    del varpool['REVIEW_RESULT']
                    
                res = self.run_program(new_code_sni,varpool,start=start)

            # program wrong
            elif status == 1:
                prompt = content.replace('ANSWER =','return')
                prompt += '\nPlease generate new code in ```python``` format.Do not use functions that have not been defined.'
                resp = self.codellm.parse(content=prompt,add_to_history=True)['content']
                new_code,new_code_sni= self.get_sni_from_text(resp)
                self.all_ori_code.append(new_code)
                self.all_code_snippets.append(new_code_sni)
                varpool = self.init_varpool.copy()
                res = self.run_program(new_code_sni,varpool)

        return res

    def step(self,codeline, varpool,index):
        #status 0 ：succes
        #status 1 : program wrong
        #status 2 : logic wrong

        #specific for VISPROG
        if 'EVAL' in codeline:
            pattern = '(?<=expr\s=)[^)]+'
            expr = re.search(pattern,codeline).group(0).strip()
            current_varpool = varpool.copy()
            for var_name,var_value in current_varpool.items():
                if isinstance(var_value,str):
                    if var_value.isdecimal():
                        current_varpool[var_name] = var_value
                    else:
                        current_varpool[var_name] = f"'{var_value}'"
                else:
                    current_varpool[var_name] = var_value
            codeline = codeline.replace(expr,expr.format(**current_varpool))
        try:
            exec(compile(codeline,'code','exec'),varpool)
        except Exception as e:
            detailed_e = traceback.format_exc()
            codeline_index = self.all_cs_map[-1][index]
            codeline=self.all_code_snippets[-2][codeline_index]
            codeline = self.cs2code(codeline)
            e = f'Error in execution of statement "{codeline}":{e}.Fix the error by generating new function startswith "def execute_command" and do not use undefined api.'
            self.logger.log(logging.INFO,str(detailed_e ))
            return 1,e
        
        if 'REVIEW_RESULT' in varpool.keys():
            r = varpool.get('REVIEW_RESULT')
            if r is not None:
                return 2,r
            
        return 0,'finish'

