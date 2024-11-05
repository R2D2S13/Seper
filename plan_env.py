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
from LLM.chatgpt_function_model import ScoreReviewFunction,TemplateChatGpt
from prompts.review_prompt import Review_Prompt_V5
import numpy as np
from vision_processes.image_patch import ImagePatch
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
from utils.utils import get_modules_functions_from_file,MyLogger
import logging

@dataclass
class PlanWrapper():
    
    config: dict = config
    function_check:bool = False
    find_check:bool = False
    simple_query_check:bool = False
    code_check:bool = False
    save_dir:str = 'results/plan_env_result'
    inject_html:bool = False
    save_var:bool = True
    log_dir:str = None
    process_id:int = None

    def __post_init__(self):
        from prompts.plan_prompts import plan_prompt_V4,outputer
        from prompts.agent_prompts import all_agent_prompt,instruct_template_V2
        self.agent_dict = {'OUTPUT':outputer}
        self.logger = MyLogger(log_dir=self.log_dir,process_id=self.process_id)
        self.all_varpool= []
        self.current_state = {}
        self.step_dict = {}
        self.agent_code_dict ={}
        self.step_max_time = 3      
        self.plan_max_time = 3 
        self.get_plan_time = 0
        self.feedback_information = {}
        self.current_previous_information = []
        self.all_previous_information = []
        self.html_str = ''
        # create plan agent
        self.plan_prompt = plan_prompt_V4
        plan_llm = TemplateChatGpt(prompt_template=self.plan_prompt,
                                   parse_args=dict(temperature=0,
                                                    top_p = 1.,
                                                    frequency_penalty=0,
                                                    presence_penalty=0))
        self.agent_dict['planer'] = plan_llm

        if not os.path.exists(self.save_dir):
            try:
                os.makedirs(self.save_dir)
            except Exception as e:
                pass

        for agent,prompt in all_agent_prompt.items():
            if agent in config.load_agents and config.load_agents[agent]:
                self.agent_dict[agent] = TemplateChatGpt(prompt_template=instruct_template_V2,
                                                         system=prompt)

        self._init_varpool={'ImagePatch':ImagePatch,'math':math,'self':self} | get_modules_functions_from_file('vision_processes/image_patch.py')| get_modules_functions_from_file('utils/inject_utils.py') 
        
    def to_json(self,data,file_name):
        folder_path = os.path.join(self.save_dir,'json')

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_name += '.json'
        file_path = os.path.join(folder_path,file_name)

        with open(file_path,'w') as f:
            json.dump(data,f,indent=2)
    
    def to_html(self,query,label,filename):
        html_save_dir = os.path.join(self.save_dir,'html')
        query = self.current_state.get('query')
        label = self.current_state.get('label')
        filename = self.current_state.get('filename')
        if not os.path.exists(html_save_dir):
            os.makedirs(html_save_dir)
        if filename is not None:
            html_file = os.path.join(html_save_dir,f'{filename}.html')
        else:
            html_file = os.path.join(html_save_dir,f'test.html')
        html_str = self.html_str
        all_code = ''
        for role,code_list in self.agent_code_dict.items():
            if len(code_list)>0:
                code = code_list[-1]
                all_code += f'\n#{role}\n{code}\n'
            else:
                all_code = 'Directly output answer'
        html_save(html_str,code=all_code,query=query,answer=label,html_file=html_file)

    def start(self,filename,query,label,**kw):
        self.all_varpool= []
        self.agent_code_dict = {}
        self.html_str = ''
        self.step_dict = {}     
        self.get_plan_time = 0
        self.current_state = dict(
            query = query,
            label = label,
            filename = filename
        )
        self.current_previous_information = []
        self.all_previous_information = []
        self.feedback_information ={}
        self.init_varpool = self._init_varpool
        for role,chat_fn in self.agent_dict.items():
            chat_fn.restart()
        self.logger.change_sample(filename)

    def inject_code(self,code):

        if self.save_var:
            code = inject_any_function(code=code,fn='self.save_intermediate_var')

        if self.inject_html:

            code = inject_html_for_whole_function(code)

        return code

    def get_info(self,inputs):
        ret = ''
        if isinstance(inputs,ImagePatch):
            ret = self.get_image_caption(inputs)
        elif isinstance(inputs,list):
            for object in inputs:
                ret += self.get_info(object) + ';'
        elif isinstance(inputs,tuple):
            for object in inputs:
                ret += self.get_info(object) + ';'
        elif inputs is None:
            ret = 'Null'
        else:
            ret = str(inputs)
        return ret

    def save_intermediate_var(self,inputs,name):

        feedback_information = self.feedback_information
        step_id = self.current_state.get('step_id')
        role = self.current_state.get('role')

        if step_id not in feedback_information.keys():
            feedback_information[step_id] = []

        info = self.get_info(inputs)

        information = f'{name}({type(inputs)}): {info}'
        feedback =  dict(
                role = role,
                feedback_information = information
            )
        repeat = False
        for current_fb in feedback_information.values():
            if len(current_fb)>0:
                for message in current_fb:
                    if information == message['feedback_information']:
                        repeat = True
                        break
        if not repeat:
            feedback_information[step_id].append(feedback)

        self.feedback_information = feedback_information


    def get_code_from_text(self,text):
        try:
            pattern=r"```(.+?)```"
            match=re.search(pattern,text,re.DOTALL)
            if match:
                code = match.group(0)
                code = code.replace('`','')
                code = code.replace('python','')
            else:
                code = text

            role = self.current_state.get('role')
            ori_code = code
            if role in self.agent_code_dict.keys():
                self.agent_code_dict[role].append(ori_code)
            else:
                self.agent_code_dict[role]=[ori_code]

            code_list = code.split('\n')
            def_indent = -1
            ret = []
            for line in code_list:
                if len(line)==0:
                    continue
                new_line = line.lstrip()
                indent = len(line) - len(new_line)
                if indent<= def_indent:
                    def_indent = -1
                if line.startswith('def execute_command'):
                    ret.append([line])
                    new_line = line.lstrip()
                    def_indent = len(line) - len(new_line)
                if not line.startswith('def execute_command') and def_indent != -1:
                    if indent > def_indent:
                        ret[-1].append(line)

            code = '\n'.join(ret[-1])
        except Exception as e:
            detailed_e = traceback.format_exc()
            self.logger.log(logging.ERROR,detailed_e)
            return 1,None,'Your function should start with "def execute_command"'

        inject_code = self.inject_code(code)

        return 0,ori_code,inject_code

    def get_image_caption(self,image):
        from vision_processes.vision_processes import forward
        if isinstance(image,ImagePatch):
            image = image.cropped_image
        caption = forward('blip',image,task='caption')
        return caption

    def get_steps_from_plan(self,text):
        steps = []
        for line in text.split('\n'):
            if line.startswith('#'):
                continue
            if 'Step' in line:
                step_id,content = line.split(':',maxsplit=1)
                args_dict = json.loads(content)
                steps.append(args_dict)
                # role:OUTPUT,args_dict:role,input_variable,output_variable
                # role:others,args_dict:role,input_variable,task_description,expected_output_result
        return steps

    def error_fix_prompt_process(self,error,role,code=None,index=None):
        if code is None:
            code = 'code'
        if role == 'planer':
            prompt = f'''- Error from previous plan running:{error}
Goal: Generate a new plan that solves the original query and avoids the above error.
'''
        else:
            prompt = f'''Error in execution of your "execute_command" function :{error}.Please correct your code,generate the new "execute_command" function in ```python``` format.
            Remeber: Only generate the new execute command function.Do not repeat previous code!'''
        return prompt

    def get_plan(self,image=None,query=None):
        # get first plan by offered image caption
        if self.get_plan_time == 0:
            caption = self.get_image_caption(image)
            caption = f'- ImageCaption : {caption}\n'
            self.current_previous_information.append(caption)
        else:
            pass
        previous_information = '\n'.join(self.current_previous_information)
        inputs = dict(
            query = query,
            previous_information = previous_information
        )
        planer = self.agent_dict.get('planer')
        plan = planer.parse(inputs,keep_history=True)[0]
        self.logger.log(logging.INFO,f'plan:{plan}')
        self.get_plan_time += 1
        return plan

    def forward(self,
                query:str,
                image_path:str,
                input_image:str,
                plan:str = None,
                file_name:str = None,
                label:str = None,
                to_json:bool = False):
        self.start(file_name,query,label)
        
        # 1.init varpool
        
        if isinstance(input_image,str):
            image = Image.open(input_image).convert('RGB')
        elif isinstance(input_image,Image.Image):
            image = input_image
            
        ImagePatch = self.init_varpool.get('ImagePatch')
        image_patch = ImagePatch(image)
        
        self.init_varpool['image_patch']= image_patch
        self.init_varpool['image'] = image
        varpool = self.init_varpool.copy()

        # 2.get firts plan
        if plan is None:
            plan = self.get_plan(image_patch.cropped_image,query)
        
        # 3.run the plan
        state,res = self.run_plan(plan,varpool)

        # 4.save the result
        if not isinstance(res,str):
            res = str(res)
        if to_json:
            assert file_name is not None
            agent_history = {}
            for role,agent in self.agent_dict.items():
                if role != 'OUTPUT':
                    agent_history[role] = agent.conversation_history[:]
            data = dict(
                    state = state,
                    query = query,
                    image_path = image_path,
                    label = label,
                    answer = res,
                    all_steps = self.step_dict,
                    feedback = self.feedback_information,
                    agent_history = agent_history
                    )

            self.to_json(data,file_name=str(file_name))
        if self.inject_html:
            self.to_html(query=query,
                         label=label,
                         filename=file_name)

        return state,res
        
    def run_plan(self,plan,varpool):
        '''
        Parameters:step type plan and varpool
        Return:result,execution state
        Run plan in multple time
        '''
    
        try:
            sub_state,res = self._run_plan(plan,varpool)
            state = 'success'
            if sub_state == 2:
                planer = self.agent_dict.get('planer')
                new_plan = planer.parse(res)[0]
                self.get_plan_time += 1
                if self.get_plan_time < self.plan_max_time:
                    state,res = self.run_plan(new_plan,varpool)
                else:
                    state = 'fail'
            elif sub_state == 1:
                state = 'fail'

        except Exception as e:
            detailed_e = traceback.format_exc()
            self.logger.log(logging.ERROR,detailed_e)
            res = str(e)
            state = 'fail'

        return state,res

    def _run_plan(self,plan,varpool,start=0):
        '''
        Run plan single time
        '''
        #status 
        # 0 ：succes
        # 1 : program wrong after 3 times fixed
        # 2 : plan wrong

        steps = self.get_steps_from_plan(plan)
        self.logger.log(logging.INFO,f'Steps:{steps}')

        if len(steps)==0:
            err = 'The plan format wrong.Generate your plan as "Steps x:your plan"'
            prompt = self.error_fix_prompt_process(err,role='planer')
            return 2,prompt
        
        for step_id,step in enumerate(steps):  
            self.current_state.update(step | {'step_id':step_id})
            role = step.get('role')
            input_variable = step.get('input_variable')
            task_description = step.get('task_description')
            expected_output_result = step.get('expected_output_result')
            output_variable = step.get('output_variable')
            if role != 'OUTPUT':
                expected_output_result_des = ';'.join([ f'{k}:{v}' for k,v in expected_output_result.items()])
                expected_output_result_var = ','.join(str(k) for k in expected_output_result.keys())

            else:
                expected_output_result_des = output_variable
                expected_output_result_var =None
            input_variable_des = ''
            if len(input_variable)>0:
            # check if the input variable given by plan llm legal or not
                try:
                    usable_variables = eval(input_variable,varpool)
                    if isinstance(usable_variables,tuple):
                        input_variable_des = ''
                        usable_vname = input_variable.split(',')
                        for v_name,v in zip(usable_vname,usable_variables):
                            input_variable_des += f'{v_name}:{type(v)};'
                    elif isinstance(usable_variables,list):
                        input_variable_des = ''
                        input_variable = input_variable.replace('[','').replace(']','')
                        usable_vname = input_variable.split(',')
                        for v_name,v in zip(usable_vname,usable_variables):
                            input_variable_des += f'{v_name}:{type(v)};'
                        input_variable_des = f'[{input_variable_des}]'

                    else:
                        input_variable_des = f'{input_variable}:{type(usable_variables)}'

                except Exception as e:
                    detailed_e = traceback.format_exc()
                    self.logger.log(logging.ERROR,detailed_e)
                    if isinstance(e,NameError):
                        prompt = self.error_fix_prompt_process(e,role='planer')
                        return 2,prompt
            else:
                input_variable_des = "image_patch:<class 'ImagePatch'>"
                input_variable = 'image_patch'
            # get the result of one step
            inputs = dict(
                usable = input_variable_des,
                query = task_description,
                varname = expected_output_result_des
            )
            msg = f'Step {step_id} inputs:{inputs}\n'
            self.logger.log(logging.INFO,msg)

            # execute step code
            status,res = self.step(inputs=inputs,
                                   input_variable=input_variable,
                                   varpool=varpool,
                                   expected_result=expected_output_result_var,
                                   role=role,
                                   step_id=step_id)
            msg = f'Step {step_id} result:{status},{res}\n'
            self.logger.log(logging.INFO,msg)
            if status!=0:
                break
            
        self.all_varpool.append(varpool)

        return status,res
    
    def step(self,inputs,input_variable,varpool,expected_result,role,step_id):
        # 0 :success
        # 1 :program wrong after 3 times fixed
        # 2 :output variable wrong after 3 times fixed
        agent = self.agent_dict.get(role)
        resp = agent.parse(inputs)[0]
        
        # OUTPUT is the final step
        if role=='OUTPUT':
            res = eval(resp,varpool)
            return 0,res
        else:
            # generate step code
            state,_,inject_code = self.get_code_from_text(resp)

            msg = f'Step {step_id} code:{inject_code}\n'
            self.logger.log(logging.INFO,msg)
            self.current_state.update(dict(
                inject_code = inject_code
            ))
            if f'Step {step_id}' in self.step_dict.keys():
                self.step_dict[f'Step {step_id}'].append(self.current_state.copy())
            else:
                self.step_dict[f'Step {step_id}'] = []
                self.step_dict[f'Step {step_id}'].append(self.current_state.copy())

            if state == 1:
                prompt = inject_code
                self.step(prompt,input_variable,varpool,expected_result,role,step_id)
                
            # res = self.run_program(inject_code,expected_result,input_variable,varpool,role,step_id)
            # return res

            try:
                # get the fn
                exec(compile(inject_code,'code','exec'),varpool)
                output_variable_codeline = f'{expected_result} = execute_command({input_variable})'
                self.logger.log(logging.INFO,f'Output variable codeline:{output_variable_codeline}')
                exec(compile(output_variable_codeline,'code','exec'),varpool)

            # code check
            except Exception as e:
                
                detailed_e = traceback.format_exc()
                prompt = self.error_fix_prompt_process(error=e,role=role)
                self.logger.log(logging.ERROR,detailed_e)
                if len(self.step_dict[f'Step {step_id}']) < self.step_max_time:
                    return self.step(prompt,input_variable,varpool,expected_result,role,step_id)
                else:
                    return 1,prompt
            
            # output variable check.TODO:Make it more flex
            try:
                eval(expected_result,varpool)
            except Exception as e:
                detailed_e = traceback.format_exc()
                prompt = 'The returned value does not meet the requirements'
                self.logger.log(logging.ERROR,detailed_e)
                if len(self.step_dict[f'Step {step_id}']) < self.step_max_time:
                    return self.step(prompt,input_variable,varpool,expected_result,role,step_id)
                else:
                    return 1,prompt

            return 0,'success'

    def run_program(self,code,expected_result,input_variable,varpool,role,step_id,**kw):

        try:
            # get the fn
            exec(compile(code,'code','exec'),varpool)
            output_variable_codeline = f'{expected_result} = execute_command({input_variable})'
            self.logger.log(logging.INFO,f'Output variable codeline:{output_variable_codeline}')
            exec(compile(output_variable_codeline,'code','exec'),varpool)

        # code check
        except Exception as e:
            
            detailed_e = traceback.format_exc()
            prompt = self.error_fix_prompt_process(error=e,role=role)
            self.logger.log(logging.ERROR,detailed_e)
            # if len(self.step_dict[f'Step {step_id}']) < self.step_max_time:
            #     return self.step(prompt,input_variable,varpool,expected_result,role,step_id)
            # else:
            #     return 1,prompt
        
        # output variable check.TODO:Make it more flex
        try:
            eval(expected_result,varpool)
        except Exception as e:
            detailed_e = traceback.format_exc()
            prompt = 'The returned value does not meet the requirements'
            self.logger.log(logging.ERROR,detailed_e)
            # if len(self.step_dict[f'Step {step_id}']) < self.step_max_time:
            #     return self.step(prompt,input_variable,varpool,expected_result,role,step_id)
            # else:
            #     return 1,prompt
            
        self.current_varpool = varpool
        
        return 0,'success'
            
    def review(self,output_variable,target,parameters,query,role):
        # 0:success
        # 1:change
        if not self.function_check:
            return 0,'success'
        if role == 'finder':
            if isinstance(output_variable,list):
                if len(output_variable)==0:
                    prompt =  f'The find function fail to find "{target}" patch.Try to use another object name with a similar meaning'
                    return 1,prompt
        if role == 'querier':
            inputs = dict(
                model_output = output_variable,
                object = parameters,
                query = query
            )
            review_result,score_result = self.review_llm.review(inputs)
            pass
         