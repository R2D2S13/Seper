Examples =[
'''
Visual Question:What item of furniture is to the right of the nightstand?
AI Answer:kitchen
#Thinking: Kitchen is not furniture.
Refuse.

Visual Question:What animal is on the bed?
AI Answer: cat
#Tninking: Cat is a type of animal
Accept.\n
''',
'''
Visual Question:Are there polar bear?
AI Answer: Yes
#Thinking: This is a true or false question, is it reasonable to answer as a true or false question.
Accept.

Visual Question:What is the color of that animal in the river?
AI Answer: Can not found animal in the river.
#Thinking: There should be an animal as the query mention.
Refuse.\n
''',
'''
Visual Question: What is the man holding?
AI Answer: He is holding a vase.
#Thinking: As an object, a vase can be held.
Accept.

Visual Question: What color do the desk have?
AI Answer: No desk found in the image.
#Thinking: There should be a desk as the query mention.
Refuse.\n
''',
'''
Visual Question:Do you think the color of the flower is red?
AI Answer:Yes.
#Thinking: Red is common color for flowers in the image.
Accept.

Visual Question:What item of furniture is beside the vase?
AI Answer:bed.
#Thinking: Of course bed is an item.And possible be placed by beside the vase.
Accept.

Visual Question:Where is the boy?
AI Answer:Park.
#Thinking: Park is clear enough as a location and does not need to be improved.
Accept.\n
''',
'''
Visual Question:What fruit is to the right of the plate?
AI Answer:No vegetable found to the left of the knife.
#Thinking: Given that the question mentions the presence of fruit in the image, it is highly probable that the image contains fruit.
Refuse.\n
''',
'''
Visual Question:How long is the road?
AI Answer:Very long.
#Thinking:Clear enough.
Accept.\n
''',
'''
Visual Question:Is there door or cabinet?
AI Answer:Yes.
#Thinking: This is a true or false question, is it reasonable to answer as a true or false question.
Accept.\n
''',
'''
Visual Question:Which kind of device is the wall in front of?
AI Answer:Computer.
#Thinking: Computer is an device.Reasonable.
Accept.\n
''',
'''
Visual Question:Where is the pants?
AI Answer:On the suitcase.
#Thinking: Pants can be placed on top of the suitcase.
Accept.\n
''',
'''
Visual Question:What is the water spinach in?
AI Answer:Noodles.
#Thinking: It could be water spinach in a bowl of noodles
Accept.\n
''',
'''
Visual Question:How big is the watch on the mans wrist?
AI Answer:Big.
#Thinking: 'Big' is a word for describe size of object.
Accept.\n
''',
'''
Visual Question:What does that man seem to be doing?
AI Answer:Batter.
#Thinking: Batter is a career,so the answer is ok.
Accept.\n
'''
]
proxy_user_prompt_V1 = f'''
Suppose you are a checker, and you are working on a visual question task with another AI, who will give his answer to the visual question,
and your task is to determine whether the answer conforms to the form of the question 
Output 'Accept' or 'Refuse' with your thinking process.
Learn from examples.
Examples as follows:
{"".join(Examples)}
(Example finished)

Remeber:

- The AI answer does not need to be extremely precise, but rather within a certain range. For example, the answer to the question “Where is here?” can be "street" or "parking lot", rather than a specific location.
- You do not need to judge the correctness of the answer! You also do not have the ability to judge it!
- Do not give 'correct answer' of the question.Becuase you can not give correct answer for an unseen image.

Visual Question:{{(question)}}
AI Answer:{{(answer)}}

'''

answer_select_prompt = '''
Imagine you are a reasoning expert in the question answering field, and you are now collaborating with another AI to answer questions. He is the answerer, 
and he thinks there are multiple possible answers to the question. What you need to do is to combine these answers semantically.

Question:
Which kind of aninamls to the right of the cabinie?
Candidate Answers:
['cats','cats','chicken']
Reasoning process:
#Thinking:The question specifies that there is only one type of animal, but there are multiple different answers in the response. Therefore, only one kind of animal in the candidate answers is correct. We can use a voting method and choose the cat with the most votes as the final answer.
Final answer:cats

Question:
Whiat tools are in the image?
Candidate Answers:
['Scissors', 'hammer','wrench','wrench', 'screwdriver']
Reasoning process:
#Thinking:The answer does not specify that there is only one type of tool, so the multiple answers in the candidate answers are not conflicting but are in parallel. The final answer should include each different tool in the candidate answers.
Final answer:The tools include scissors, hammer, wrench, screwdriver.

Question:
{(query)}
Candidate Answers:
{(candidate_answers)}
Reasoning process:
#Thinking:

'''

analyse_prompt_V1 = f'''
You are a code engineer who is good at understanding the state of running code from intermediate variables and can analyze the correctness of function execution in conjunction with functions.
You are now collaborating with another AI to generate a code that can solve a vision problem, and then his code doesn't output a reasonable answer. 
You need to analyze his code and the intermediate variables generated during code execution and analyze what the cause is

Examples as follows:
(Example 1)
Code:
# What is the color of the carpet?
def execute_command(image):
    image_patch = ImagePatch(image)
    carpet_patch = image_patch.find('carpet')
    if len(carpet_patch)==0:
        return 'unknown'
    else:
        return carpet_patch[0].simple_query('What is the color?')

The return value of code:
unknown

The return value unreasonable reason:
unknown is ambiguous and devoid of any meaning

Intermediate variable:
...
if len(carpet_patch)==0:True
...

Analyse:
The reason why return unkown is that:from the middle state variable 'if len(carpet_patch)==0:True'.It can be found that find method can not find carpet in the image so that 
the final return value is 'unknown'.
Suggestion:
1.Explore more parameters.Replace the carpet in the "carpet = image_patch.find('carpet')" with other words of the same semantics
2.Add simple_query function for error handler.If the find method return an empty list,direct use simple_query method to get answer like:
    "if len(capert_patch)==0:
        return image_patch.simple_query(question)"
(Example 1 finished)

(Example 2)
Code:
# Is the red jackect long sleeved or short sleeved?
def execute_command(image):
    image_patch = ImagePatch(image)
    red_jackect_patch = image_patch.find('red_jackect')
    is_short_sleeved = red_jackect_patch.verify_property("blue shirt", "short sleeved")
    is_long_sleeved = red_jackect_patch.verify_property("blue shirt", "long sleeved")
    if is_short_sleeved:
        return "short sleeved"
    elif is_long_sleeved:
        return "long sleeved"
    else:
        return "unknown"
The return value of code:
unknown

The return value unreasonable reason:
unknown is ambiguous and devoid of any meaning

Intermediate variable:
is_short_sleeved:False
is_long_sleeved:False
if is_short_sleeved?:False
elif is_long_sleeved?:False

Analyse:
The reason why return unkown is that:The code assume that one of the "is_short_sleeved" and "is_long_sleeved" variable will be True,but it didn't.
Both the variable "is_short_sleeved" and "is_long_sleeved" is generated by 'verify_property' methods.
However, both attributes judged by this method are false, resulting in an incorrect final answer.
Suggestion:
1.Use best_text_match method is better than verify_property method.The formmer will absolutely return an answer during 2 options.
2.Add simple_query function for error handler.If both ,direct use simple_query method to get answer like:
    "else:
        return image_patch.simple_query(question)"
(Example 2 finished)

(Example 3)
Code:
# How big is the sign on the building?
def execute_command(image)->str:
    image_patch = ImagePatch(image)
    clock_patches = image_patch.find("clock")
    # Question assumes only one clock patch
    if len(clock_patches) == 0:
        # If no clock is found, query the image directly
        return image_patch.simple_query("How big is the clock on the wall?")
    clock_patch = clock_patches[0]
    if clock_patch.width > clock_patch.height:
        return "wide"
    elif clock_patch.width < clock_patch.height:
        return "tall"
    else:
        return "square"
The return value of code:
tall

The return value unreasonable reason:
Tall is not a kind of size.

Intermediate variable:
...
elif clock_patch.width < clock_patch.height:True
...

Analyse:
The reason why return tall is that:The code attempting to determine size by comparing width and length is unreasonable, and it is impossible to determine size through pixel distance of an image.
Suggestion:
1.Use best_text_match with giving options like 'big','small',etc.
2.Use simple_query method to directly get the size.
(Example 3 finished)

(Example 4)
Code:
# What color do you think the pants the man is in is?
def execute_command(image)->str:
    image_patch = ImagePatch(image)
    man_patches = image_patch.find("man")
    # Question assumes one man patch   
    if len(man_patches) == 0:       
    # If no man is found, query the image directly        
        return image_patch.simple_query("What color do you think the pants the man is in is?")   
    man_patch = man_patches[0]    
    pants_patches = image_patch.find("pants")   
    # Question assumes one pants patch    
    if len(pants_patches) == 0:        
        return "unknown"    
    for pants_patch in pants_patches:        
        if pants_patch.horizontal_center == man_patch.horizontal_center and pants_patch.vertical_center == man_patch.vertical_center:            
            return pants_patch.best_text_match(["red", "blue", "green", "black", "white", "yellow", "orange", "purple", "pink", "brown", "gray"])
     return "unknown""

The return value of code:
unknown

The return value unreasonable reason:
unknown is ambiguous and devoid of any meaning     

Intermediate variable:
- image_patch: a kitchen with hardwood floors and a chandelier
- man_patches: lens:1
- len(man_patches): 1
- if len(man_patches) == 0?: False
- man_patch: a man holding a ski pole
- pants_patches: lens:4
- len(pants_patches): 4
- if len(pants_patches) == 0?: False
- pants_patch.horizontal_center: 190.0
- man_patch.horizontal_center and pants_patch.vertical_center: 176.0
- man_patch.vertical_center: 168.0
- if pants_patch.horizontal_center == man_patch.horizontal_center and pants_patch.vertical_center == man_patch.vertical_center?: False

Analysis:
The code attempts to find a man and a pants in the given image and determine the color of the pants the man is wearing. However, based on the intermediate variables, it seems that the code is unable to find a matching pants for the man. 
The code checks if there are any man patches found in the image. In this case, len(man_patches) is equal to 1, which means a man patch is found. Then, it checks if there are any pants patches found in the image. len(pants_patches) is equal to 4, which means there are pants patches found. 
However, the for loop that follows checks if the center coordinates of the pants patch match the center coordinates of the man patch. In this case, pants_patch.horizontal_center is 190.0, man_patch.horizontal_center is 176.0, and man_patch.vertical_center is 168.0. Based on these coordinates, the condition in the if statement is False, indicating that no matching pants is found for the man.
Therefore, the code returns "unknown" as the final answer, which is unreasonable because it should have been able to determine the color of the pants.
Suggestion:
1.The strict requirement of horizontal_center being identical is too stringent, which leads to the failure of matching pants with man. I suggest using more relaxed matching methods such as overlaps_with, for example, whether the pants are under man’s patch, or using verify_property to inquire whether the man’s attributes contain pants.
2.Moreover, if the relaxed criteria mentioned above also fail, you can directly use simple_query to inquire about mans_patch as an error handling measure.
(Example 4 finished)

(Example 5)
Code:
# What do the vase and the paper have in common?"
def execute_command(image)->str:
    image_patch = ImagePatch(image)
    vase_patches = image_patch.find("vase")
    paper_patches = image_patch.find("paper")
    # Question assumes only one vase and one paper patch
    if len(car_patches) == 0 or len(scooter_patches) == 0:
        # If no vase or paper is found, query the image directly
        return image_patch.simple_query("What do the vase and the paper have in common?")
    vase_patch = vase_patches[0]
    paper_patch = paper_patches[0]
    if vase_patch.horizontal_center == paper_patch.horizontal_center:
        return "They are in the same position."
    elif vase_patch.vertical_center == paper_patch.vertical_center:
        return "They are at the same height."
    else:
        return "They have no commonalities."

Intermediate variable:
...
- len(vase_patches)==2
- len(paper_patches)==0
...

Analysis:
It can be seem from the intermediate steps that the code successfully found the vase,but it fail to find the paper.
It can be concluded that find method can not detect 'paper' in this image.
Suggestion:
1.Try use more other synonym object name replace 'paper' in the paper_patches = image_patch.find("paper") for increase chance to find paper in the image.
2.Use the simple_query to ask if you can't find paper or vase, instead of when you can't find both
(Example 5 finished)

(Example 6)
Code:
# Is the bottole that looks full or empty?
def execute_command(image)->str:
    image_patch = ImagePatch(image)
    bottle_patches = image_patch.find("bottle")
    # Question assumes only one bottle patch
    if len(bottle_patches) == 0:
        # If no bottle is found, query the image directly
        return image_patch.simple_query("Is the bottle that looks red full or empty?")
    bottle_patch = bottle_patches[0]
    # Assuming the verify_property method checks if the bottle is full
    is_full = bottle_patch.verify_property("bottle", "full")
    # It is unclear why the code determines the bottle is full and returns "yes" as the answer
    # More information about the verify_property method and its reasoning should be provided
    return bool_to_yesno(is_full)

The return value of code:
yes

The return value unreasonable reason:
The question answer is whether the bottle is full or empty.It can not be answer by a single yes.

Intermediate variable:
image_patch:woman holding an umbrella,
bottle_patches: lens:3,
len(bottle_patches):3,
if len(bottle_patches) == 0?: False
is_full:True

Analysis:
As can be seen from the intermediate variables, the is_full variable has been judged to be true, that is, the water cup in the figure is full,
but the code uses bool_to_yes_no at the end, so the output is yes or no, not full or empty, so the code should directly output full if the is full is
full otherwise output empty
Suggestion:
1.Return 'full' or 'empty' but not 'yes' and 'no'
(Example 6 finished)

(Example 7)
Code:
# How are the vehicles to the right of the walking person that is walking on the sidewalk called?
def execute_command(image)->str:
    image_patch = ImagePatch(image)
    person_patches = image_patch.find("person")
    # Question assumes only one person patch
    if len(person_patches) == 0:
        # If no person is found, query the image directly
        return image_patch.simple_query("How are the vehicles to the right of the walking person that is walking on the sidewalk called?")
    person_patch = person_patches[0]
    vehicle_patches = image_patch.find("vehicle")
    vehicle_names = []
    for vehicle_patch in vehicle_patches:
        if vehicle_patch.horizontal_center > person_patch.horizontal_center:
            vehicle_names.append(vehicle_patch.simple_query("What is this?"))
    return ", ".join(vehicle_names)

The return value of code:
''

The return value unreasonable reason:
The ai answer is missing.

Intermediate variable:
- image_patch: a man is riding a skateboard on the side of a ramp
- person_patches: lens:2
- len(person_patches): 2
- if len(person_patches) == 0?: False
- person_patch: people walking down the street
- vehicle_patches: lens:0
- len(vehicle_patches)": 0
- if len(vehicle_patches) == 0?: True

Analysis:
The reason why the code didn't output a reasonable answer is find methods can not find vehicle patches in the image. 
The code tries to find vehicles to the right of the walking person, but since there are no vehicle patches, it cannot provide a valid answer.
Suggestion:
1.Explore more parameters.Replace the vehicle in the "car = image_patch.find('car')" with other words of the same semantics
2.Add simple_query function for error handler.If the find method return an empty list,direct use simple_query method to get answer like:
    "if len(vehicle_patch)==0:
        return image_patch.simple_query(question)"
(Example 7 finished)


Remeber:
1.Do not just repeat the analysis in the examples!
2.The return value must be a reasonable answer to a question, not "unkown","i dont know", etc
3.Just give analyse and suggestion.Do not generate code.

Code:
# {{(query)}}
{{(last_code)}}
The return value of code:
{{(last_answer)}}
The return value unreasonable reason:
{{(reason}}
Intermediate variable:
{{(interRes)}}
Analyse:

'''

feedback_prompt_V1 = f'''
The return value of your previous generated function:"{{(last_answer)}}" can not give a reasonable or specific answer,
Each line of code Exectuin result(Intermediate result):
{{(interRes)}}

Question 1:Base on the Intermediate result,which line of code becomes a precondition for your last function return {{(last_answer)}}"'

Question 2:
Some of the api's provided to you may be limited by the model resulting in less than expected results, these api's include:
find(object_name): different names for the same object may lead to different results.
simple_query(question): for different questions with the same semantics, may lead to different results.
verify_property(object_name,property):for different object_name or property with same semantics, may lead to different results.
Does line of code from question 1 contain an intermediate variable that is generated by API and that satisfies a precondition that causes the function to generate an incorrect answer?
If so, are these intermediate variables being restricted by API limitations that result in wrong answers/empty lists being returned?

Question 3:Based on the results of the analysis in the previous two steps.Generate a new function,the code needs to avoid meeting the preconditions for the function to generate the wrong answer "{{(last_answer)}}".

Here are some samples for correct:
Query:What is the color of the car behind the tree?
Reason:The find function can not find cars 
New code:
def execute_command(image):
    image = ImagePatch(image)
    image.simple_query('What is the color of the car behind the tree?')
(Examples finished)
'''

feedback_prompt_V2 = f'''
The return value of your previous generated function:"ANSWER_INSERT_HERE" can not give a reasonable answer,
The reason is REASON_INSERT_HERE
Code:
CODE_INSERT_HERE
Code Exectuin Intermediate Result:
FB_INSERT_HERE

1.Which statement in the code caused the answer to fail when analyzing with intermediate variables?
2.Extract experience from the above analysis, generate a new function, and avoid using the code that caused the answer to fail previously.
'''