import json
import sys
sys.path.append('/.do')
import openai
from openai.error import RateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
) 
from termcolor import colored
import time
import re
from utils.utils import find_last_number
import logging
from configs import config

# import openai
# #crazy mode
    
@retry(
     retry=retry_if_exception_type((openai.error.APIError,openai.error.RateLimitError,openai.error.APIConnectionError,openai.error.ServiceUnavailableError, openai.error.Timeout)), 
     wait=wait_random_exponential(multiplier=1,min=5,max=60), 
     stop=stop_after_attempt(10))
def chat_completion_request(key, messages, functions=None,function_call=None,key_pos=None, model="gpt-3.5-turbo-16k-0613",stop=None,process_id=0,random_key:bool=config.key_pooling, **args):
    use_messages = []
    for message in messages:
        if not("valid" in message.keys() and message["valid"] == False):
            use_messages.append(message)

    json_data = {
        "model": model,
        "messages": use_messages,
        "max_tokens": 1024,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        **args
    }
    if stop is not None:
        json_data.update({"stop": stop})
    if functions is not None:
        json_data.update({"functions": functions})
    if function_call is not None:
        json_data.update({"function_call": function_call})
    
    try:
        if model in ("gpt-3.5-turbo-16k-0613","gpt-3.5-turbo"):
            if random_key or key==None:
                from utils.key_utils import get_random_key
                key = get_random_key()
                openai.api_key = key
                print(f'use key:{key}')
        else:
            raise NotImplementedError
        openai_response = openai.ChatCompletion.create(
            **json_data,
        )
        json_data = json.loads(str(openai_response))
        return json_data 


    except Exception as e:
        if isinstance(e,openai.error.RateLimitError):
            if '(RPD)' in str(e) or 'exceeded your current quota' in str(e):
                if random_key or key==None:
                    from utils.key_utils import delete_key
                    print(f'delete key:{key}')
                    delete_key(key)
        print("Unable to generate ChatCompletion response")
        print(f"OpenAI calling Exception: {e}")
        raise e
        # return e



class PureChatGpt():

    def __init__(self, model="gpt-3.5-turbo-16k-0613",openai_key="",parse_args = None,system = None,logger=None):
        self.model = model
        self.conversation_history = []
        self.openai_key = openai_key
        self.time = time.time()
        self.TRY_TIME = 2
        self.parse_args = parse_args
        self.system = system
        self.logger = logger
        if system is not None:
            self.add_message({'role':'system','content':system})

    def add_message(self, message):
        self.conversation_history.append(message)

    def restart(self):
        self.conversation_history = []
        if self.system is not None:
            self.add_message({'role':'system','content':self.system})

    def change_messages(self,messages):
        self.conversation_history = messages

    def display_conversation(self, detailed=False):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        print("before_print"+"*"*50)
        for message in self.conversation_history:
            print_obj = f"{message['role']}: {message['content']} "
            if "function_call" in message.keys():
                print_obj = print_obj + f"function_call: {message['function_call']}"
            print_obj += ""
            print(
                colored(
                    print_obj,
                    role_to_color[message["role"]],
                )
            )
        print("end_print"+"*"*50)

    def _change_key(self):
        pass

    def parse(self, content = None,key_pos=None,add_to_history=True,**args):
        conversation_history = self.conversation_history
        logger = self.logger
        if self.parse_args:
            args = self.parse_args | args

        if content is not None:
            user_message = {'role':'user','content':content}
            conversation_history.append(user_message)

        for t in range(self.TRY_TIME):
            if t != 0:
                time.sleep(15)

            json_data = chat_completion_request(
                self.openai_key, conversation_history,key_pos=key_pos, **args
            )

            try:
                total_tokens = json_data['usage']['total_tokens']
                if 'n' in args.keys() and args['n']>1: 
                    #not generate only one time
                    message_list = []
                    for choice in json_data["choices"]:

                        message = choice["message"]
                        message_list.append(message)

                    return message_list #return message and do not add to history
                
                else:
                    message = json_data["choices"][0]['message']
                if add_to_history:
                    self.add_message(message)

                if "function_call" in message.keys() and "." in message["function_call"]["name"]:
                    message["function_call"]["name"] = message["function_call"]["name"].split(".")[-1]

                return message
            
            except RateLimitError as e:
                print(str(e))
        return e

    def _send(self,agent,content):
        '''send and get response of another agent'''

        assert isinstance(agent,PureChatGpt),"You must specify an agent"

        for _ in range(self.TRY_TIME):
            if _ != 0:
                time.sleep(15)

            try: 
                message = agent.parse(content)
                # message['role'] = 'user' #所有的agent本体都是assistant,接受user的信息
                return message
            
            except BaseException as e:
                print(f"Parsing Exception: {repr(e)}. Try again.")

        return e
    
    def initiate_chat(self,agent,start,turns,stop=None):
        '''receieve start message and send message after parsing'''
        # assert self.system is not None
        message = self.parse(content=start)['content']
        # if stop is None:
        for _ in range(turns):
            try:
                resp = self._send(content=message,agent=agent)['content']
                print('user:' + resp)
                time.sleep(10)
                message = self.parse(resp)['content']
                print('assistant:' + message)
                if stop is not None:
                    if stop in message:
                        print(stop)
                        break

            except Exception as e:
                return f'OPENAI_ERROR:{e}'
                
        return self.conversation_history
                
class CheckFunction(PureChatGpt):
    def __init__(self, model="gpt-3.5-turbo-16k-0613", openai_key="", parse_args=None, system=None,prompt_template = None,threshold=3.0,logger = None):
        super().__init__(model, openai_key, parse_args, system,logger)
        self.pattern = r"\{([^}]*)\}"
        self.prompt_template = prompt_template
        if self.parse_args is None:
            self.parse_args = {'n':3}
        self.threshold = threshold

    def review(self,inputs):
        #used fix prompt
        prompt = self.prompt_template
        matchs = re.findall(pattern=self.pattern,string=prompt)
        
        #put args in prompt
        for match in matchs:
            var = str(inputs.get(match))
            prompt = prompt.replace(f'{{{match}}}',var)

        #get_res
        check_res = self.parse(content=prompt) 
        self.conversation_history = []
        return check_res 

class ScoreReviewFunction(PureChatGpt):
    def __init__(self, model="gpt-3.5-turbo-16k-0613", openai_key="", parse_args=None, system=None,prompt_template = None,threshold=3.0,logger = None):
        super().__init__(model, openai_key, parse_args, system,logger)
        self.pattern = r"\{([^}]*)\}"
        self.prompt_template = prompt_template
        if self.parse_args is None:
            self.parse_args = {'n':3}
        self.threshold = threshold

    def review(self,inputs):
        review_result = None
        #used fix prompt
        prompt = self.prompt_template
        matchs = re.findall(pattern=self.pattern,string=prompt)
        
        #put args in prompt
        for match in matchs:
            var = str(inputs.get(match))
            prompt = prompt.replace(f'{{{match}}}',var)

        #get_res
        review_result = self.parse(content=prompt)

        score = self.get_score(review_result)
        if self.logger is not None:
            self.logger.log(logging.INFO,f'get score {score}')
        # #score < threshold
        # if score < self.threshold:
        #     review_result = score_result
        #     # review_result = score_result[0].split('\n')[1]#get the reasong process.
            
        #clear history
        self.conversation_history=[]
        
        return review_result,score
        
        # return score_result
    
    def parse(self, content = None,key_pos=None,add_to_history=True,**args)->list:
        logger = self.logger
        conversation_history = self.conversation_history

        if self.parse_args:
            args = self.parse_args | args

        if content is not None:
            user_message = {'role':'user','content':content}
            conversation_history.append(user_message)

        for t in range(self.TRY_TIME):
            if t != 0:
                time.sleep(15)

            json_data = chat_completion_request(
                self.openai_key, conversation_history,key_pos=key_pos, **args
            )

            try:
                total_tokens = json_data['usage']['total_tokens']
                if 'n' in args.keys() and args['n']>1: 
                    #not generate only one time
                    message_list = []
                    for choice in json_data["choices"]:

                        message = choice["message"]
                        message_list.append(message['content'])

                    return message_list #return message and do not add to history
                
                else:
                    message = json_data["choices"][0]['message']
                if add_to_history:
                    self.add_message(message)

                if "function_call" in message.keys() and "." in message["function_call"]["name"]:
                    message["function_call"]["name"] = message["function_call"]["name"].split(".")[-1]

                return [message['content']]
            
            except RateLimitError as e:
                if isinstance(e,RateLimitError):
                    k = self._change_key()
                    logger.log(logging.WARNING,f'{e}.change self.openai_key to {k}.')
                    t -= 1
                if isinstance(e,TimeoutError):
                    logger.log(logging.WARNING,f'{e}.TimeOut')
                    t -= 1
                logger.log(logging.WARNING,f"Parsing Exception: {repr(e)}. Try again.")
        
        return e

    def get_score(self,review_res,**args):
        score = 0
        for res in review_res:
            score += find_last_number(res)
        n = self.parse_args['n']
        score = score/n
        return score
    
class TemplateChatGpt(PureChatGpt):

    def __init__(self, model="gpt-3.5-turbo-16k-0613", openai_key="", parse_args=None, system=None,prompt_template = None,logger = None):
        super().__init__(model, openai_key, parse_args, system,logger)
        self.pattern = r"\{\(([^}]*)\)\}"#pattern:{()}
        self.prompt_template = prompt_template

    def parse(self,inputs:dict or str,keep_history = True,**kw)->list[str]:
        #used fix prompt
        prompt = self.prompt_template

        # Fill dict values in template
        if isinstance(inputs,dict):
            matchs = re.findall(pattern=self.pattern,string=prompt)
            #put args in prompt
            for match in matchs:
                var = str(inputs.get(match))
                prompt = prompt.replace(f'{{({match})}}',var)#same with pattern
        # Do not fill in template
        else:
            prompt=str(inputs)

        result = self._parse(content=prompt,**kw)
        #clear history
        if not keep_history:
            self.restart()
        
        return result
        
        # return score_result
    
    def _parse(self, content = None,key_pos=None,add_to_history=True,**args) -> list[str]:
        logger = self.logger
        conversation_history = self.conversation_history

        if self.parse_args:
            args = self.parse_args | args

        if content is not None:
            user_message = {'role':'user','content':content}
            conversation_history.append(user_message)

        for t in range(self.TRY_TIME):
            if t != 0:
                time.sleep(15)

            json_data = chat_completion_request(
                self.openai_key, conversation_history,key_pos=key_pos, **args
            )

            try:
                total_tokens = json_data['usage']['total_tokens']
                if 'n' in args.keys() and args['n']>1: 
                    #not generate only one time
                    message_list = []
                    for choice in json_data["choices"]:

                        message = choice["message"]
                        message_list.append(message['content'])

                    return message_list #return message and do not add to history
                
                else:
                    message = json_data["choices"][0]['message']
                if add_to_history:
                    self.add_message(message)

                if "function_call" in message.keys() and "." in message["function_call"]["name"]:
                    message["function_call"]["name"] = message["function_call"]["name"].split(".")[-1]

                return [message['content']]
            
            except RateLimitError as e:
                if isinstance(e,RateLimitError):
                    k = self._change_key()
                    if logger is not None:
                        logger.log(logging.WARNING,f'{e}.change self.openai_key to {k}.')
                    t -= 1
                if isinstance(e,TimeoutError):
                    if logger is not None:
                        logger.log(logging.WARNING,f'{e}.TimeOut')
                    t -= 1
                if logger is not None:
                    logger.log(logging.WARNING,f"Parsing Exception: {repr(e)}. Try again.")
        
        return None

class LevelReviewFunction(TemplateChatGpt):
    def __init__(self, model="gpt-3.5-turbo-16k-0613", openai_key="", parse_args=None, system=None, prompt_template=None, logger=None,negative_threshold=2):
        super().__init__(model, openai_key, parse_args, system, prompt_template, logger)
        self.negative_threshold = negative_threshold

    def parse(self, inputs: dict, keep_history=True) -> list[str]:
        review_result = super().parse(inputs, keep_history)
        n = 0
        for v in review_result:
            v1,v2 = v.split('\n')
            v1 = v1.split('?')[1].strip().replace(':','')
            v2 = v2.split('?')[1].strip().replace(':','')
            if 'Impossible' in v1 or 'impossible' in v1:
                # print(v1)
                last_reason = v1
                n += 1
            if v2.startswith('No') or v2.startswith('no'):
                # print(v2)
                last_reason = v2
                n += 1

        if n >= self.negative_threshold:
            # print(id)
            if len(last_reason.split('.'))>2:
                last_reason = last_reason.split('.')[1]
            elif len(last_reason.split(','))>2:
                last_reason = last_reason.split(',')[1]
            return last_reason
        else:
            return None

class LevelReviewFunctionForOKVQA(TemplateChatGpt):
    def __init__(self, model="gpt-3.5-turbo-16k-0613", openai_key="", parse_args=None, system=None, prompt_template=None, logger=None,negative_threshold=2):
        super().__init__(model, openai_key, parse_args, system, prompt_template, logger)
        self.negative_threshold = negative_threshold

    def parse(self, inputs: dict, keep_history=True,return_thinking_process:bool = False) -> list[str]:
        review_result = super().parse(inputs, keep_history)
        n = 0
        for v in review_result:
            try:
                judge = v.split('judge:')[-1].strip()
            except Exception as e:
                judge = v
            judge = judge.lower()
            if 'impossible' in judge:
                # print(v1)
                try:
                    thinking_process = v
                    last_reason = judge.split('impossible')[-1]
                except Exception as e:
                    thinking_process = v
                    last_reason = judge
                n += 1

        if n >= self.negative_threshold:
            # print(id)
            if len(last_reason.split('.'))>2:
                last_reason = last_reason.split('.')[1]
            elif len(last_reason.split(','))>2:
                last_reason = last_reason.split(',')[1]
            if return_thinking_process:
                return thinking_process,last_reason
            else:
                return last_reason
        else:
            if return_thinking_process:
                return None,None
            else:
                return None
        

class FindReviewFunction(TemplateChatGpt):
    def __init__(self, model="gpt-3.5-turbo-16k-0613", openai_key="", parse_args=None, system=None, prompt_template=None, logger=None,negative_threshold = 1):
        super().__init__(model, openai_key, parse_args, system, prompt_template, logger)
        self.dict_extract_pattern = r"\{([^}]+)\}"
        self.negative_threshold = negative_threshold
        self.fix_prompt = '''Please generate response follow  json format like
        Output:{'existence':...,
                'possible_location':...} 
        '''
        self.extract_max_try_time = 3
        self.extract_try_time = 0

    def parse(self, inputs: dict, keep_history=True) -> list[str]:
        res = super().parse(inputs)
        n = 0

        for v in res:
            print(v)
            match = re.search(self.dict_extract_pattern,v)

            if match is None:
                print('GOT NONE MATCH')
                if self.extract_try_time<self.extract_max_try_time:
                    self.extract_try_time += 1
                    return self.parse(self.fix_prompt)
                else:
                    self.extract_try_time = 0
                    if not keep_history:
                        self.conversation_history = []
                    return None
            
            match = match.group()
            match = eval(match)
            print(match)
            d = list(match.values())
            try:
                existence,location = d[0],d[1]
            except Exception as e:
                if self.extract_try_time<self.extract_max_try_time:
                    self.extract_try_time += 1
                    return self.parse(self.fix_prompt)
                else:
                    self.extract_try_time = 0
                    if not keep_history:
                        self.conversation_history = []
                    return None
                
            existence = existence.lower()
            if 'impossible' not in existence:
                n += 1
                last_location = location.lower()

        if not keep_history:
            self.conversation_history = []

        self.extract_try_time = 0
        
        if n >= self.negative_threshold:
            if last_location == 'unknown':
                last_location = "Can't get a definite position" 
            return last_location
        else:
            return None
            

class FinalReviewFunction(TemplateChatGpt):
    def __init__(self, model="gpt-3.5-turbo-16k-0613", openai_key="", parse_args=None, system=None, prompt_template=None, logger=None):
        super().__init__(model, openai_key, parse_args, system, prompt_template, logger)
    def review(self,inputs):
        res = self.parse(inputs)
        last_reason = None
        an = 0
        rn = 0
        for r in res:
            r = r.lower()
            if 'accept' in r:
                an += 1
            elif 'refuse' in r:
                last_reason = r.split('.',maxsplit=1)[1]
                rn += 1
        if an > rn:
            accept_or_not = True
        else:
            accept_or_not = False
        return last_reason,accept_or_not
