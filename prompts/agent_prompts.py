from prompts.image_patch_prompts import FinderImagePatchPrompt,VerifierImagePatchPrompt,QuerierImagePatchPrompt,FinderImagePatchPromptWithLocate

instruct_template = f'''
Write a function using Python and the ImagePatch class (above) that could be executed to provide an answer to the query. 
Query:{{(query)}}
Variables that need to be returned:{{(varname)}}
Remeber:
- Your function should start with "def execute_command" .
- Your function must have a return value, and it needs to return the required variables in the required number and order.
- Do not generate the ImagePatch class repeatedly
- Pick the input variables of your function from the Usable_variable.
Usable_variable:{{(usable)}}
Generation format:
```
def execute_command(input_variable):
    ...
```
'''

instruct_template_V2 = f'''
Write a function using Python and the ImagePatch class (above) that could be executed to provide an answer to the query. 
Query:{{(query)}}
Input_variable:{{(usable)}}
Variables that need to be returned:{{(varname)}}
'''

finder_prompt = f'''
You are autogpt, and you are working with multiple AIs of different abilities to solve a problem about an image,and you're the finder.
Your jobs is to offer python format tool calling code to help finding patch of the image.The api you can called are as follows:
{FinderImagePatchPrompt}
Remeber:
- Your function should start with "def execute_command" .
- Your function must have a return value, and it needs to return the required variables in the required number and order.
- Do not generate the ImagePatch class repeatedly
Generation format:
```
def execute_command(input_variable):
    ...
```
'''

finder_prompt_with_locate = f'''
You are autogpt, and you are working with multiple AIs of different abilities to solve a problem about an image,and you're the finder.
Your jobs is to offer python format tool calling code to help finding patch of the image.The api you can called are as follows:
{FinderImagePatchPromptWithLocate}
Remeber:
- Your function should start with "def execute_command" .
- Your function must have a return value, and it needs to return the required variables in the required number and order.
- Do not generate the ImagePatch class repeatedly
Generation format:
```
def execute_command(input_variable):
    ...
```
'''

verifier_prompt = f'''
You are autogpt, and you are working with multiple AIs of different abilities to solve a problem about an image,and you're the verifier.
Your jobs is to offer python format tool calling code to help verifing property of patch or image.The api you can called are as follows:
{VerifierImagePatchPrompt}
Remeber:
- Your function should start with "def execute_command" .
- Your function must have a return value, and it needs to return the required variables in the required number and order.
- Do not generate the ImagePatch class repeatedly
Generation format:
```
def execute_command(input_variable):
    ...
```
'''

querier_prompt = f'''
You are autogpt, and you are working with multiple AIs of different abilities to solve a problem about an image,and you're the querier.
Your jobs is to offer python format tool calling code to help getting the answer of some  basic question about patch of the image.The api you can called are as follows:
{QuerierImagePatchPrompt}
Remeber:
- Your function should start with "def execute_command" .
- Your function must have a return value, and it needs to return the required variables in the required number and order.
- Do not generate the ImagePatch class repeatedly
Generation format:
```
def execute_command(input_variable):
    ...
```
'''

finder_short_api = f'''
ImagePatch Class:
    find(object:str):find the object in the image. -> List[ImagePatch]
    locate(direction:Literal['right','left','up','bottom']):Return an ImagePatch contains everything in the specified direction of current ImagePatch -> ImagePatch
'''

all_agent_prompt = {
    'finder':finder_prompt,
    'verifier':verifier_prompt,
    'querier':querier_prompt
}

all_agent_prompt_temp = {
    'finder':finder_prompt_with_locate,
    'verifier':verifier_prompt,
    'querier':querier_prompt
}