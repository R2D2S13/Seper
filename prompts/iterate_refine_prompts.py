intermediate_analyse_prompt = '''
I want you to serve as a senior code engineer. I will provide you with the code,including its intermediate states and return values. 
First.You should be able to trace back from the return values to identify the root cause statements that lead to these return values. Combining with other variables, you need to analyze why this return value occurs.
Second.Summarize the content of the intermediate variables. The term 'image_patch' represents the content of the image; the number of patches indicates how many of a certain object are present in the image.
Third.Analyse from questions code and intermediate results.Try to find possible problems.

Examples as follows:
(Example1)
code:
def execute_command(image)->str:
    image_patch = ImagePatch(image)
    person_patches = image_patch.find("person")
    # Question assumes only one person patch
    if len(person_patches) == 0:
        # If no person is found, query the image directly
        return image_patch.simple_query("What does the person to the left of the helmet hold?")
    person_patch = person_patches[0]
    helmet_patches = image_patch.find("helmet")
    # Question assumes only one helmet patch
    if len(helmet_patches) == 0:
        return "no"
    for helmet_patch in helmet_patches:
        if helmet_patch.horizontal_center > person_patch.horizontal_center:
            return helmet_patch.simple_query("What does the person to the left of the helmet hold?")
    return "no"
intermediate_result:
    'image_patch': 'a baseball game',
    'person_patches': 'lens:4',
    'len(person_patches)': '4',
    'if len(person_patches) == 0?': 'False',
    'person_patch': 'a baseball player holding a bat',
    'helmet_patches': 'lens:3',
    'len(helmet_patches)': '3',
    'if len(helmet_patches) == 0?': 'False',
    'helmet_patch.horizontal_center': '333.5',
    'person_patch.horizontal_center': '230.0',
    'if helmet_patch.horizontal_center > person_patch.horizontal_center?': 'True'
return value: right

{
    'Return value traceability analysis': The root line is the "helmet_patch.simple_query("What does the person to the left of the helmet hold?")" line.The reason is
    because horizontal_center of one of the helmet_patch bigger than that of person_patch.
    'Summary of intermediate variables': The image shows a baseball game,there are 4 persons in the image and 3 helmet.
    'Possible problems': The code assume that there are only one person patch,but actually there are 4 persons.The code is missing the handling of different situations when the number of people is greater than 1.
}

(Example2)
# Which color is the device the monitor is to the right of?
def execute_command(image)->str:
    image_patch = ImagePatch(image)
    monitor_patches = image_patch.find("monitor")
    # Question assumes only one monitor patch
    if len(monitor_patches) == 0:
        # If no monitor is found, query the image directly
        return image_patch.simple_query("Which color is the device the monitor is to the right of?")
    for monitor_patch in monitor_patches:
        device_patches = image_patch.find("device")
        for device_patch in device_patches:
            if device_patch.horizontal_center < monitor_patch.horizontal_center:
                return device_patch.simple_query("Which color is the device?")
    return "unknown"

intermediate_result:
    "image_patch": "a computer monitor and a laptop computer on a desk",
      "monitor_patches":"lens:2",
      "len(monitor_patches)":"2",
      "if len(monitor_patches) == 0?":"False",
      "device_patches":"lens:4",
      "device_patch.horizontal_center":"285.0",
      "monitor_patch.horizontal_center": "319.0",
      "if device_patch.horizontal_center < monitor_patch.horizontal_center?": "True"
return value: right

{
    'Return value traceability analysis': The root line is the "return device_patch.simple_query("Which color is the device?")" line.The reason is
    because horizontal_center of one of the device_patch smaller than that of monitor_patch.
    'Summary of intermediate variables': The image shows a computer monitor and a laptop computer on a desk,There are 2 monitor and 4 devices.
    'Possible problems': Although there are multiple Monitors in the problem, for the device, the judgment condition is already satisfied at the first device, and the answer is returned without obtaining the colors of other devices that meet the condition later.
}

(Example4)
# How big is the donkey to the right of the other donkeys??
def execute_command(image)->str:
    image_patch = ImagePatch(image)
    donkey_patches = image_patch.find("donkey")
    # Question assumes at least two donkey patches
    if len(donkey_patches) < 2:
        # If less than two donkeys are found, query the image directly
        return image_patch.simple_query("How big is the donkey to the right of the other donkeys?")
    for i in range(len(donkey_patches)):
        for j in range(i+1, len(donkey_patches)):
            if donkey_patches[i].horizontal_center < donkey_patches[j].horizontal_center:
                return "smaller"
    return "larger"

intermediate_result:
      "image_patch":"a herd of donkeys in a field",
      "donkey_patches":"lens:3",
      "len(donkey_patches)":"3",
      "if len(donkey_patches) < 2?":"False",
      "donkey_patches[i].horizontal_center":"125.5",
      "donkey_patches[j].horizontal_center":"316.5",
      "if donkey_patches[i].horizontal_center < donkey_patches[j].horizontal_center?":"True"
      
return value: smaller

{
    'Return value traceability analysis': The root line is the "if donkey_patches[i].horizontal_center < donkey_patches[j].horizontal_center:" line.The reason is
    because horizontal_center of one of the donkey smaller than that of donkey_patch,which means one of the donkey is more left than other donkey.
    'Summary of intermediate variables': The image shows a a herd of donkeys in a field,there are 3 donkeys at different position.
    'Possible problems': It is not reasonable to try to judge the size by comparing the horizontal_center, and the code should be based on the area, not the position of the axis in the patch, which will only indicate that one zebra is closer to the left than the others
}

(Example5)
# In which part of the photo is the bridge, the bottom or the top?
def execute_command(image)->str:
    image_patch = ImagePatch(image)
    bridge_patches = image_patch.find("bridge")
    # Question assumes only one bridge patch
    if len(bridge_patches) == 0:
        # If no bridge is found, query the image directly
        return image_patch.simple_query("In which part of the photo is the bridge, the bottom or the top?")
    if bridge_patches[0].vertical_center > image_patch.vertical_center:
        return "bottom"
    else:
        return "top"

intermediate_result:
image_patch:A bridge cross the river.
bridge_patches:lens:1
if bridge_patches[0].vertical_center > image_patch.vertical_center? True
...
      
return value: bottom

{
    'Return value traceability analysis': The root line is the "if bridge_patches[0].vertical_center > image_patch.vertical_center:" line.The reason is
    because the vertical_center of the bridge is bigger than that of image_patch.
    'Summary of intermediate variables': The image shows 1 bridge cross the river.
    'Possible problems': The judgment of up and down in the code is reversed. The patch with a larger vertical center should be placed above, and similarly, the patch with a larger horizontal center should be placed on the right side of the image.
}

(Example6)
# Are there any black pants or scarves?
def execute_command(image)->str:
    image_patch = ImagePatch(image)
    is_black_pant = image_patch.exists("black pant")
    is_scarf = image_patch.exists("scarf")
    return bool_to_yesno(is_black_pant or is_scarf)

intermediate_result:
image_patch:A man playing football.
is_black_pant:False
is_scarf:False
...
      
return value: bottom

{
    'Return value traceability analysis': The root line is the "bool_to_yesno(is_black_pant or is_scarf)" line.The reason is
    because both is_black and is_scarf are false.
    'Summary of intermediate variables': The image shows man playing football but no black pants and scarf.
    'Possible problems': No problem in the code.
}
(Examples finished)

The first code is:{(code)}
The intermediate result:{(inter_res)}
return_value:{(return_value)}

Your return format should be as follows:
{
    'Return value traceability analysis': Which line of code is the root reason of return {(return_value)}?
    'Summary of intermediate variables': Included length of list and text description.
    'Possible problems': The possible problems hide in code based on query and summary of intermediate variables.
}
'''

summry_prompt = '''
I want you to serve as a senior code engineer. I will provide you with the code,including its intermediate states and return values. 
First.You should be able to trace back from the return values to identify the root cause statements that lead to these return values. Combining with other variables, you need to analyze why this return value occurs.
Second.Summarize the content of the intermediate variables. Combining the statistical results of variables with their corresponding statements and summarizing them.
Example as follows:
(Example 1)
# What is used to make the container on the floor, wood or metal?
Code:
def execute_command(image)->str:
    image_patch = ImagePatch(image)
    container_patches = image_patch.find("container")
    # Question assumes only one container patch
    if len(container_patches) == 0:
        # If no container is found, query the image directly
        return image_patch.simple_query("What is used to make the container on the floor, wood or metal?")
    for container_patch in container_patches:
        is_wood = container_patch.verify_property("container", "wood")
        is_metal = container_patch.verify_property("container", "metal")
        if is_wood:
            return "wood"
        elif is_metal:
            return "metal"
    # If no container is found, query the image directly
    return image_patch.simple_query("What is used to make the container on the floor, wood or metal?")

Intermediate results:
image_patch:a white toilet and the shape (c,h,w) of image_patch:torch.Size([3, 375, 500])
container_patches:lens:2
len(container_patches):2
if len(container_patches) == 0?:False
is_wood?: False
The second time of is_wood: False
is_metal?:False
The second time of is_metal:True

Summary: 
The image shows a white toilet,the 'find' methods detect 2 container patches.The firtst container is neither wood or metal.
The variables is_wood and is_metal are both generated by the 'verify_property' method. 
In the evaluation of the second container patch, where is_metal is True, meeting the conditions of the elif is_metal, the final return value is 'metal'.
(Example finished)


Code:
{(code)}
Intermediate result:
{(inter_res)}
return_value:
{(return_value)}

Summary:

'''