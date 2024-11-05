

tool_des=[
"find_objects(image:Image,object_name:str or list) -> Image - Returns a list of Image objects matching object_name contained in the crop if any are found.\n",
"knowledge_retrieve(question:str) -> str - Access external information and answer informational questions not concerning the image,return the answer.\n",
"pos_relation(image1:Image,image2:Image) -> tuple(str) - Returns the positional relationship of the first picture to the second picture.Return value in the tuple include left,right,top,bottom,front,behind,inside,overlap.such as:(left,front),(right,inside)",
"color_judge(image:Image) -> str- Return the color of a image.\n",
"verify_property(image:Image,property:str)-> bool - Returns True if the object possesses the visual property, and False otherwise.\n",
"get_property(image:Image,property:str) -> str - Returns the visual property (e.g., shape, material) of an image"
]

AI_des = [
"finder(image or objects:str) -> find the object in the image\n",
"verifier(objects: list or str,property: list or str) -> verify whether object possesses the visual property\n",
"querier(objects: list or str,query: str)-> answer to a basic question asked about the object or image.\n",
"captioner(objects:list or str) -> get the image caption from image or object\n",
"reasoner(context:str,question:str) -> Reasoning from the text based on context or common sense\n"
]

AI_des_V2 = [
"finder(image or objects:str) -> find the object in the image,return patch of image\n",
"verifier(objects: list or str) -> verify whether object possesses the visual property,return str\n",
"querier(objects: list or str)-> answer to a basic question asked about the object or image or get their property,return str.\n",
"captioner(objects:list or str) -> get the image caption from image or object,return str\n",
"reasoner() -> Reasoning from the text based on context or common sense,return str\n",
"OUTPUT()-> Output the final answer.Should follow the python format\n"
]

Examples_V2=[
'''
Information from previous run: 
- ImageCaption : a zebra standing next to a giraffe

Goal: Answer the query "What type of animal is beautiful, the deer or the zebra?"

# Think:From the information provided by captioner, it can be confirmed that there are zebras and deer in the image.I need to verify if the zebras and deer are both beautiful and first need to find the zebras and deer.
Step 1:finder(image): find all zebra and deer in the image -> zebra_patch:The zebra in the image,deer_patch:The deer in the image
# Think:Now i have found the certain zebra and deer,i need to verify wheteher they are beautiful.
Step 2:verifier([zebra_patch,deer_patch]): verify whether the zebra or the deer is beautiful ->zebra_beautiful:A bool value indicate whteher the zebra is beautiful,deer_beautiful:A bool value indicate whteher the deer is beautiful
# Think:Now i have the beautiful property of the deer and zebra,i just need to output the answer.
Step 3:OUTPUT(deer_beautiful,zebra_beautiful):f"The deer is {deer_beautiful} and the zebra is {zebra_beautiful}"


Information from previous run:
- ImageCaption : a zebra standing next to a giraffe
- The final answer of your previous plan is "both zebra and deer is beautiful",which did not answer the query.
- The object you have found :[zebra,deer]

Goal: Use another way to get the answer of query.
# Think:Since beauty is a subjective issue that can't be solved by objective means, perhaps I should get the answer to my question by common sense extrapolation
Step 1:reasoner(): Based on the context,who is prettier,a zebra or a deer?You must choice one -> answer:The answer of who is more beautiful in context
Step 2:OUTPUT(answer):answer
''',
'''
Information from previous run: 
- ImageCaption : a laptop computer on the ground next to a bag

Goal: Answer the query "Does the book have the same color as the charger?"

# Think:First i should find the book and charger in the image
Step 1:finder(image):find all book and charger in the image -> book_patch:The book in the image,charger_patch:The charger in the image
# Think:Now i have found the book and charger,i need to verify wheteher they have same color
Step 2:verifier([book_patch,charger_patch]):Verify that the color is the same between any two of the book and the charger ->color_same_or_not:A bool value indicate whteher the color is same or not
Step 3:OUTPUT(color_same_or_not):color_same_or_not
''',
'''
Information from previous run: 
- ImageCaption : a man and a child sitting on a fire hydrant in front of a building

Goal: Answer the query "Is the man to the right or to the left of the kid?"

# Think:First i should find the kid.I can tell from the caption that there is only one kid.
Step 1:finder(image):find the kid -> kid_patch:The kid in the image
# Think:Then i should find the man near the kid
Step 2:finder(image):find the man next to the kid -> man_patch:The man next to the kid
# Think:At last,i need to judge the position relation between the kid and the man,i will call querier to complete this task.
Step 3:querier([kid_patch,man_patch]):Is the man to the right or to the left of the kid? -> answer:The answer of the query
Step 4:OUTPUT(answer):answer
''',
'''
Information from previous run: 
- ImageCaption : a woman laying on a bed

Goal: Answer the query "Is this a table or a bed?"

# Think:I can get the answer directly from the caption, as it hints at a bed in the picture
Step 1:OUTPUT():f"Yes,there is a bed."
''',
'''
Information from previous run: 
- ImageCaption : a black curtain behind the desk

Goal: Answer the query "What is in front of the curtain?"

# Think: The caption mentions that the curtain is behind the table, and this curtain is obviously the one in the query, so the answer is that the table is in front of the curtain.
Step 1:OUTPUT():f"desk."
''',
'''
Information from previous run: 
- ImageCaption : a fire hydrant is on the side of a building

Goal: Answer the query "Are there any mirrors or cars?"
# Think:I need to call finder to find if there are mirrors or cars in the image
Step 1:finder(image): find mirrors and cars -> mirrors_patch:The mirrors in the image, cars_patch:The car in the image
# Think:Considering that only one of the mirror and the car needs to exist,I'll use or to connect them.
Step 2:OUTPUT(mirrors_patch or cars_patch):mirrors_patch or cars_patch
''',
'''
Information from previous run: 
- ImageCaption : a cat laying on a glass desk

Goal: Answer the query "Is the dirt white and light?"

# Think:Since the caption doesn't mention the dirt attribute, I need to find the dirt patch with the find tool, and then let the verifier identify it as white and light
# Think:I need to call finder to find the dirt at first
Step 1:finder(image): find all dirt in this image -> dirt_patch:The dirt in the image
# Think:Call verifier to verify the property of the dirt.As long as there is one dirt_patch that meets the requirements.So i should use 'any' in the statement of sub-tasks.
Step 2:verifier(dirt_patch):verify if any of the dirt is white and light -> dirt_white_and_light_or_not:whether the dirt is white and light or not,bool value
# Think:Now i just need to output the verified result
Step 3:OUTPUT(dirt_white_and_light_or_not):dirt_white_and_light_or_not
''',
'''
Information from previous run: 
- ImageCaption : a black and white photo of a kitchen stove

Goal: Answer the query "What kind of appliance is the bowl sitting on top of?"

# Think:The caption implies that the scene in the picture is a kitchen, so the appliances should also be common kitchen appliances
# Think:In order to find the appliance under the bowl, I first needed to find the bowl
Step 1:finder(image): find the bowl -> bowl_patch:The bowl in the image
# Think:Next i should find the appliance under the bowl
Step 2:finder(bowl_patch): find the kitchen appliance under the bowl -> appliance_patch:The appliance under the bowl
# Think:I'll let querier tell me what kind of appliance it is.
Step 3:querier(appliance_patch):What kind of appliance is this?->appliance_type:The type of appliance
Step 4:OUTPUT(appliance_type):appliance_type
''',
'''
Information from previous run: 
- ImageCaption : a table with a vase of flowers on it
Goal: Answer the query "On which side is the bulb?"
# Think:At first i should find the bulb
Step 1:finder(image): find the bulb -> bulb_patch:The bulb in the image
# Think:Next i should get the bulb position
Step 2:querier(bulb_patch):On which side is the bulb?->bulb_side:The side of the bulb in this image
Step 3:OUTPUT(bulb_side):f"The bulb is on the {bulb_side} side of the vase."
''',
'''
Information from previous run: 
- ImageCaption : a table with a vase of flowers on it
Goal: Answer the query "On which side is the bulb?"
# Think:At first i should find the bulb
Step 1:finder(image): find the bulb -> bulb_patch:The bulb in the image
# Think:Next i should get the bulb position
Step 2:querier(bulb_patch):On which side is the bulb?->bulb_side:The side of the bulb in this image
Step 3:OUTPUT(bulb_side):f"The bulb is on the {bulb_side} side of the vase."
''',
]
Examples_V3=[
'''
Information from previous run: 
- ImageCaption : a zebra standing next to a giraffe

Goal: Answer the query "What type of animal is beautiful, the deer or the zebra?"

# Think:Can i get the answer directyly from the previous information?
# Think:No.The caption only mentions the presence of zebras and giraffes, but neither the presence of Deer, nor the presence of Deer, nor who is more beautiful, so I need to make a detailed plan.

# Think:From the information provided by captioner, it can be confirmed that there are zebras and deer in the image.I need to verify if the zebras and deer are both beautiful and first need to find the zebras and deer.
Step 1:finder(image): find all zebra and deer in the image -> zebra_patch:The zebra in the image;deer_patch:The deer in the image
# Think:Now i have found the certain zebra and deer,i need to verify wheteher they are beautiful.
Step 2:verifier([zebra_patch,deer_patch]): verify whether the zebra or the deer is beautiful ->zebra_beautiful:A bool value indicate whteher the zebra is beautiful;deer_beautiful:A bool value indicate whteher the deer is beautiful
# Think:Now i have the beautiful property of the deer and zebra,i just need to output the answer.
Step 3:OUTPUT(deer_beautiful,zebra_beautiful):f"The deer is {deer_beautiful} and the zebra is {zebra_beautiful}"


Information from previous run:
- ImageCaption : a zebra standing next to a giraffe
- The final answer of your previous plan is "both zebra and deer is beautiful",which did not answer the query.
- The object you have found :[zebra,deer]

Goal: Use another way to get the answer of query.
# Think:Can i get the answer directyly from the previous information?
# Think:No.The failure of my previous plan was mentioned in the previous plan.Since beauty is a subjective issue that can't be solved by objective means, perhaps I should get the answer to my question by common sense extrapolation

Step1:reasoner: Based on the context,who is prettier,a zebra or a deer?You must choice one -> answer:The answer of who is more beautiful in context
Step2:OUTPUT(answer):answer
''',
'''
Information from previous run: 
- ImageCaption : a laptop computer on the ground next to a bag

Goal: Answer the query "Does the book have the same color as the charger?"

# Think:Can i get the answer directyly from the previous information?
# Think:No.Because there is no mention of the color of the book and charger in the caption.

# Think:First i should find the book and charger in the image
Step 1:finder(image):find all book and charger in the image -> book_patch:The book in the image;charger_patch:The charger in the image
# Think:Now i have found the book and charger,i need to verify wheteher they have same color
Step 2:verifier([book_patch,charger_patch]):Verify that the color is the same between any two of the book and the charger ->color_same_or_not:A bool value indicate whteher the color is same or not
Step 3:OUTPUT(color_same_or_not):color_same_or_not
''',
'''
Information from previous run: 
- ImageCaption : a man and a child sitting on a fire hydrant in front of a building

Goal: Answer the query "Is the man to the right or to the left of the kid?"
# Think:Can i get the answer directyly from the previous information?
# Think:No.Previous information only mentions the existence of man and kid, and does not mention their positional relationship.I need to make a detailed plan.

# Think:First i should find the kid.I can tell from the caption that there is only one kid.
Step 1:finder(image):find the kid -> kid_patch:The kid in the image
# Think:Then i should find the man near the kid
Step 2:finder(image):find the man next to the kid -> man_patch:The man next to the kid
# Think:At last,i need to judge the position relation between the kid and the man,i will call querier to complete this task.
Step 3:querier([kid_patch,man_patch]):Is the man to the right or to the left of the kid? -> answer:The answer of the query
Step 4:OUTPUT(answer):answer
''',
'''
Information from previous run: 
- ImageCaption : a woman laying on a bed

Goal: Answer the query "Is this a table or a bed?"
# Think:Can i get the answer directyly from the previous information?
# Think:Yes.I can get the answer directly from the caption, as it hints at a bed in the picture

Step 1:OUTPUT():f"Yes,there is a bed."
''',
'''
Information from previous run: 
- ImageCaption : a black curtain behind the desk

Goal: Answer the query "What is in front of the curtain?"
# Think: Can i get the answer directyly from the previous information?
# Think: Yes.The caption mentions that the curtain is behind the table, and this curtain is obviously the one in the query, so the answer is that the table is in front of the curtain.
Step 1:OUTPUT():f"desk."
''',
'''
Information from previous run: 
- ImageCaption : a fire hydrant is on the side of a building

Goal: Answer the query "Are there any mirrors or cars?"
# Think: Can i get the answer directyly from the previous information?
# Think: No.Previous information doesn't mention the presence of mirrors or cars, but that doesn't necessarily mean they're not in the image.

# Think:I need to call finder to find if there are mirrors or cars in the image
Step 1:finder(image): find mirrors and cars -> mirrors_patch:The mirrors in the image;cars_patch:The car in the image
# Think:Considering that only one of the mirror and the car needs to exist,I'll use or to connect them.
Step 2:OUTPUT(mirrors_patch or cars_patch):mirrors_patch or cars_patch
''',
'''
Information from previous run: 
- ImageCaption : a cat laying on a glass desk

Goal: Answer the query "Is the dirt white and light?"
# Think: Can i get the answer directyly from the previous information?
# Think: No.There is no information about dirt in the image caption, and I can't reason directly about the answer from the caption.

# Think:Since the caption doesn't mention the dirt attribute, I need to find the dirt patch with the find tool, and then let the verifier identify it as white and light
# Think:I need to call finder to find the dirt at first

Step 1:finder(image): find all dirt in this image -> dirt_patch:The dirt in the image
# Think:Call verifier to verify the property of the dirt.As long as there is one dirt_patch that meets the requirements.So i should use 'any' in the statement of sub-tasks.
Step 2:verifier(dirt_patch):verify if any of the dirt is white and light -> dirt_white_and_light_or_not:whether the dirt is white and light or not
# Think:Now i just need to output the verified result
Step 3:OUTPUT(dirt_white_and_light_or_not):dirt_white_and_light_or_not
''',
'''
Information from previous run: 
- ImageCaption : a black and white photo of a kitchen stove

Goal: Answer the query "What kind of appliance is the bowl sitting on top of?"
# Think: Can i get the answer directyly from the previous information?
# Think: No.But the caption implies that the scene in the picture is a kitchen, so the appliances should also be common kitchen appliances

# Think:In order to find the appliance under the bowl, I first needed to find the bowl
Step 1:finder(image): find the bowl -> bowl_patch:The bowl in the image
# Think:Next i should find the appliance under the bowl
Step 2:finder(bowl_patch): find the kitchen appliance under the bowl -> appliance_patch:The appliance under the bowl
# Think:I'll let querier tell me what kind of appliance it is.
Step 3:querier(appliance_patch):What kind of appliance is this?->appliance_type:The type of appliance
Step 4:OUTPUT(appliance_type):appliance_type
''',
'''
Information from previous run: 
- ImageCaption : a table with a vase of flowers on it
Goal: Answer the query "On which side is the bulb?"
# Think: Can i get the answer directyly from the previous information?
# Think: No.Because it does not mention the side of the bulb.I need to make a plan.

# Think:At first i should find the bulb
Step 1:finder(image): find the bulb -> bulb_patch:The bulb in the image
# Think:Next i should get the bulb position
Step 2:querier(bulb_patch):On which side is the bulb?->bulb_side:The side of the bulb in this image
Step 3:OUTPUT(bulb_side):f"The bulb is on the {bulb_side} side of the vase."
''',
]


Examples_V3_dict=[
'''
Information from previous run: 
- ImageCaption : a zebra standing next to a giraffe

Goal: Answer the query "What type of animal is beautiful, the deer or the zebra?"

# Think:Can i get the answer directyly from the previous information?
# Think:No.The caption only mentions the presence of zebras and giraffes, but neither the presence of Deer, nor the presence of Deer, nor who is more beautiful, so I need to make a detailed plan.

# Think:From the information provided by captioner, it can be confirmed that there are zebras and deer in the image.I need to verify if the zebras and deer are both beautiful and first need to find the zebras and deer.
Step 1:finder(image): find all zebra and deer in the image -> zebra_patch:The zebra in the image;deer_patch:The deer in the image
# Think:Now i have found the certain zebra and deer,i need to verify wheteher they are beautiful.
Step 2:verifier([zebra_patch,deer_patch]): verify whether the zebra or the deer is beautiful ->zebra_beautiful:A bool value indicate whteher the zebra is beautiful;deer_beautiful:A bool value indicate whteher the deer is beautiful
# Think:Now i have the beautiful property of the deer and zebra,i just need to output the answer.
Step 3:OUTPUT(deer_beautiful,zebra_beautiful):f"The deer is {deer_beautiful} and the zebra is {zebra_beautiful}"


Information from previous run:
- ImageCaption : a zebra standing next to a giraffe
- The final answer of your previous plan is "both zebra and deer is beautiful",which did not answer the query.
- The object you have found :[zebra,deer]

Goal: Use another way to get the answer of query.
# Think:Can i get the answer directyly from the previous information?
# Think:No.The failure of my previous plan was mentioned in the previous plan.Since beauty is a subjective issue that can't be solved by objective means, perhaps I should get the answer to my question by common sense extrapolation

Step 1:reasoner(): Based on the context,who is prettier,a zebra or a deer?You must choice one -> answer:The answer of who is more beautiful in context
Step 2:OUTPUT(answer):answer
''',
'''
Information from previous run: 
- ImageCaption : a laptop computer on the ground next to a bag

Goal: Answer the query "Does the book have the same color as the charger?"

# Think:Can i get the answer directyly from the previous information?
# Think:No.Because there is no mention of the color of the book and charger in the caption.

# Think:First i should find the book and charger in the image
Step 1:finder(image):find all book and charger in the image -> book_patch:The book in the image;charger_patch:The charger in the image
# Think:Now i have found the book and charger,i need to verify wheteher they have same color
Step 2:verifier([book_patch,charger_patch]):Verify that the color is the same between any two of the book and the charger ->color_same_or_not:A bool value indicate whteher the color is same or not
Step 3:OUTPUT(color_same_or_not):color_same_or_not
''',
'''
Information from previous run: 
- ImageCaption : a man and a child sitting on a fire hydrant in front of a building

Goal: Answer the query "Is the man to the right or to the left of the kid?"
# Think:Can i get the answer directyly from the previous information?
# Think:No.Previous information only mentions the existence of man and kid, and does not mention their positional relationship.I need to make a detailed plan.

# Think:First i should find the kid.I can tell from the caption that there is only one kid.
Step 1:finder(image):find the kid -> kid_patch:The kid in the image
# Think:Then i should find the man near the kid
Step 2:finder(image):find the man next to the kid -> man_patch:The man next to the kid
# Think:At last,i need to judge the position relation between the kid and the man,i will call querier to complete this task.
Step 3:querier([kid_patch,man_patch]):Is the man to the right or to the left of the kid? -> answer:The answer of the query
Step 4:OUTPUT(answer):answer
''',
'''
Information from previous run: 
- ImageCaption : a woman laying on a bed

Goal: Answer the query "Is this a table or a bed?"
# Think:Can i get the answer directyly from the previous information?
# Think:Yes.I can get the answer directly from the caption, as it hints at a bed in the picture

Step 1:OUTPUT():f'Yes,there is a bed.'
''',
'''
Information from previous run: 
- ImageCaption : a black curtain behind the desk

Goal: Answer the query "What is in front of the curtain?"
# Think: Can i get the answer directyly from the previous information?
# Think: Yes.The caption mentions that the curtain is behind the table, and this curtain is obviously the one in the query, so the answer is that the table is in front of the curtain.
Step 1:OUTPUT():f'desk.'
''',
'''
Information from previous run: 
- ImageCaption : a fire hydrant is on the side of a building

Goal: Answer the query "Are there any mirrors or cars?"
# Think: Can i get the answer directyly from the previous information?
# Think: No.Previous information doesn't mention the presence of mirrors or cars, but that doesn't necessarily mean they're not in the image.

# Think:I need to call finder to find if there are mirrors or cars in the image
Step 1:finder(image): find mirrors and cars -> mirrors_patch:The mirrors in the image;cars_patch:The car in the image
# Think:Considering that only one of the mirror and the car needs to exist,I'll use or to connect them.
Step 2:OUTPUT(mirrors_patch or cars_patch):mirrors_patch or cars_patch
''',
'''
Information from previous run: 
- ImageCaption : a cat laying on a glass desk

Goal: Answer the query "Is the dirt white and light?"
# Think: Can i get the answer directyly from the previous information?
# Think: No.There is no information about dirt in the image caption, and I can't reason directly about the answer from the caption.

# Think:Since the caption doesn't mention the dirt attribute, I need to find the dirt patch with the find tool, and then let the verifier identify it as white and light
# Think:I need to call finder to find the dirt at first

Step 1:finder(image): find all dirt in this image -> dirt_patch:The dirt in the image
# Think:Call verifier to verify the property of the dirt.As long as there is one dirt_patch that meets the requirements.So i should use 'any' in the statement of sub-tasks.
Step 2:verifier(dirt_patch):verify if any of the dirt is white and light -> dirt_white_and_light_or_not:whether the dirt is white and light or not
# Think:Now i just need to output the verified result
Step 3:OUTPUT(dirt_white_and_light_or_not):dirt_white_and_light_or_not
''',
'''
Information from previous run: 
- ImageCaption : a black and white photo of a kitchen stove

Goal: Answer the query "What kind of appliance is the bowl sitting on top of?"
# Think: Can i get the answer directyly from the previous information?
# Think: No.But the caption implies that the scene in the picture is a kitchen, so the appliances should also be common kitchen appliances

# Think:In order to find the appliance under the bowl, I first needed to find the bowl
Step 1:finder(image): find the bowl -> bowl_patch:The bowl in the image
# Think:Next i should find the appliance under the bowl
Step 2:finder(bowl_patch): find the kitchen appliance under the bowl -> appliance_patch:The appliance under the bowl
# Think:I'll let querier tell me what kind of appliance it is.
Step 3:querier(appliance_patch):What kind of appliance is this?->appliance_type:The type of appliance
Step 4:OUTPUT(appliance_type):appliance_type
''',
'''
Information from previous run: 
- ImageCaption : a table with a vase of flowers on it
Goal: Answer the query "On which side is the bulb?"
# Think: Can i get the answer directyly from the previous information?
# Think: No.Because it does not mention the side of the bulb.I need to make a plan.

# Think:At first i should find the bulb
Step 1:finder(image): find the bulb -> bulb_patch:The bulb in the image
# Think:Next i should get the bulb position
Step 2:querier(bulb_patch):On which side is the bulb?->bulb_side:The side of the bulb in this image
Step 3:OUTPUT(bulb_side):f'The bulb is on the {bulb_side} side of the vase.'
''',
]


plan_prompt_V2=f'''
You are autogpt, and you are working with multiple AIs of different abilities to solve a problem about an image,and you're the planner.
First you get information about the image as a whole provided by the captioner,then write an abstract plan to successfully answer the query. 
In each step of the plan mention which AI (including sub-tasks and conditions) that need to be called. Learn from and incorporate information from previous runs.

You can call the following AIs:\n
{"".join(AI_des_V2)}
The AI call should follow the format like: AI(object):subtask -> varname

Here are examples:
{"".join([f'(Example{index} start):{example}(Example{index} finished)' for index,example in enumerate(Examples_V3)])}
(All Examples finished)

Here is a new query.First you get information about the image as a whole provided by the captioner,then write an abstract plan to successfully answer the query. 
In each step of the plan mention which AI (including sub-tasks and conditions) that need to be called. Learn from and incorporate information from previous runs.
Remeber:
1.Follow the format of the python code when using the output function at the end.
2.Try to infer more information from the image caption first in your thinking process, and then make a plan/

Information from previous run: {{previous_information}}
Goal: Answer the query "{{query}}"
# Think: Can i get the answer directyly from the previous information?

'''
Examples_V4='''
(Example 0 start):
Information from previous run: 
- ImageCaption : a zebra standing next to a giraffe

Goal: Answer the query "What type of animal is beautiful, the deer or the zebra?"

# Think:Can i get the answer directyly from the previous information?
# Think:No.The caption only mentions the presence of zebras and giraffes, but neither the presence of Deer, nor the presence of Deer, nor who is more beautiful, so I need to make a detailed plan.

# Think:From the information provided by captioner, it can be confirmed that there are zebras and deer in the image.I need to verify if the zebras and deer are both beautiful and first need to find the zebras and deer.
Step 1:{"role": "finder", "input_variable": "image", "task_description": "find all zebra and deer in the image ", "expected_output_result": {"zebra_patch": "The zebra in the image", "deer_patch": "The deer in the image"}}
# Think:Now i have found the certain zebra and deer,i need to verify wheteher they are beautiful.
Step 2:{"role": "verifier", "input_variable": "[zebra_patch,deer_patch]", "task_description": "verify whether the zebra or the deer is beautiful ", "expected_output_result": {"zebra_beautiful": "A bool value indicate whteher the zebra is beautiful", "deer_beautiful": "A bool value indicate whteher the deer is beautiful"}}
# Think:Now i have the beautiful property of the deer and zebra,i just need to output the answer.
Step 3:{"role": "OUTPUT", "input_variable": "deer_beautiful,zebra_beautiful", "output_variable": "The deer is {deer_beautiful} and the zebra is {zebra_beautiful}"}


Information from previous run:
- ImageCaption : a zebra standing next to a giraffe
- The final answer of your previous plan is "both zebra and deer is beautiful",which did not answer the query.
- The object you have found :[zebra,deer]

Goal: Use another way to get the answer of query.
# Think:Can i get the answer directyly from the previous information?
# Think:No.The failure of my previous plan was mentioned in the previous plan.Since beauty is a subjective issue that can't be solved by objective means, perhaps I should get the answer to my question by common sense extrapolation

Step 1:{"role": "reasoner", "input_variable": "", "task_description": "Based on the context,who is prettier,a zebra or a deer?You must choice one ", "expected_output_result": {"answer": "The answer of who is more beautiful in context"}}
Step 2:{"role": "OUTPUT", "input_variable": "answer", "output_variable": "answer"}
(Example 0 finished)
(Example 1 start):
Information from previous run: 
- ImageCaption : a laptop computer on the ground next to a bag

Goal: Answer the query "Does the book have the same color as the charger?"

# Think:Can i get the answer directyly from the previous information?
# Think:No.Because there is no mention of the color of the book and charger in the caption.

# Think:First i should find the book and charger in the image
Step 1:{"role": "finder", "input_variable": "image", "task_description": "find all book and charger in the image ", "expected_output_result": {"book_patch": "The book in the image", "charger_patch": "The charger in the image"}}
# Think:Now i have found the book and charger,i need to verify wheteher they have same color
Step 2:{"role": "verifier", "input_variable": "[book_patch,charger_patch]", "task_description": "Verify that the color is the same between any two of the book and the charger ", "expected_output_result": {"color_same_or_not": "A bool value indicate whteher the color is same or not"}}
Step 3:{"role": "OUTPUT", "input_variable": "color_same_or_not", "output_variable": "color_same_or_not"}
(Example 1 finished)
(Example 2 start):
Information from previous run: 
- ImageCaption : a man and a child sitting on a fire hydrant in front of a building

Goal: Answer the query "Is the man to the right or to the left of the kid?"
# Think:Can i get the answer directyly from the previous information?
# Think:No.Previous information only mentions the existence of man and kid, and does not mention their positional relationship.I need to make a detailed plan.

# Think:First i should find the kid.I can tell from the caption that there is only one kid.
Step 1:{"role": "finder", "input_variable": "image", "task_description": "find the kid ", "expected_output_result": {"kid_patch": "The kid in the image"}}
# Think:Then i should find the man near the kid
Step 2:{"role": "finder", "input_variable": "image", "task_description": "find the man next to the kid ", "expected_output_result": {"man_patch": "The man next to the kid"}}
# Think:At last,i need to judge the position relation between the kid and the man,i will call querier to complete this task.
Step 3:{"role": "querier", "input_variable": "[kid_patch,man_patch]", "task_description": "Is the man to the right or to the left of the kid? ", "expected_output_result": {"answer": "The answer of the query"}}
Step 4:{"role": "OUTPUT", "input_variable": "answer", "output_variable": "answer"}
(Example 2 finished)
(Example 3 start):
Information from previous run: 
- ImageCaption : a woman laying on a bed

Goal: Answer the query "Is this a table or a bed?"
# Think:Can i get the answer directyly from the previous information?
# Think:Yes.I can get the answer directly from the caption, as it hints at a bed in the picture

Step 1:{"role": "OUTPUT", "input_variable": "", "output_variable": "Yes,there is a bed."}
(Example 3 finished)
(Example 4 start):
Information from previous run: 
- ImageCaption : a black curtain behind the desk

Goal: Answer the query "What is in front of the curtain?"
# Think: Can i get the answer directyly from the previous information?
# Think: Yes.The caption mentions that the curtain is behind the table, and this curtain is obviously the one in the query, so the answer is that the table is in front of the curtain.
Step 1:{"role": "OUTPUT", "input_variable": "", "output_variable": "desk"}
(Example 4 finished)
(Example 5 start):
Information from previous run: 
- ImageCaption : a fire hydrant is on the side of a building

Goal: Answer the query "Are there any mirrors or cars?"
# Think: Can i get the answer directyly from the previous information?
# Think: No.Previous information doesn't mention the presence of mirrors or cars, but that doesn't necessarily mean they're not in the image.

# Think:I need to call finder to find if there are mirrors or cars in the image
Step 1:{"role": "finder", "input_variable": "image", "task_description": "find mirrors and cars ", "expected_output_result": {"mirrors_patch": "The mirrors in the image", "cars_patch": "The car in the image"}}
# Think:Considering that only one of the mirror and the car needs to exist,I'll use or to connect them.
Step 2:{"role": "OUTPUT", "input_variable": "mirrors_patch or cars_patch", "output_variable": "mirrors_patch or cars_patch"}
(Example 5 finished)
(Example 6 start):
Information from previous run: 
- ImageCaption : a cat laying on a glass desk

Goal: Answer the query "Is the dirt white and light?"
# Think: Can i get the answer directyly from the previous information?
# Think: No.There is no information about dirt in the image caption, and I can't reason directly about the answer from the caption.

# Think:Since the caption doesn't mention the dirt attribute, I need to find the dirt patch with the find tool, and then let the verifier identify it as white and light
# Think:I need to call finder to find the dirt at first

Step 1:{"role": "finder", "input_variable": "image", "task_description": "find all dirt in this image ", "expected_output_result": {"dirt_patch": "The dirt in the image"}}
# Think:Call verifier to verify the property of the dirt.As long as there is one dirt_patch that meets the requirements.So i should use 'any' in the statement of sub-tasks.
Step 2:{"role": "verifier", "input_variable": "dirt_patch", "task_description": "verify if any of the dirt is white and light ", "expected_output_result": {"dirt_white_and_light_or_not": "whether the dirt is white and light or not"}}
# Think:Now i just need to output the verified result
Step 3:{"role": "OUTPUT", "input_variable": "dirt_white_and_light_or_not", "output_variable": "dirt_white_and_light_or_not"}
(Example 6 finished)
(Example 7 start):
Information from previous run: 
- ImageCaption : a black and white photo of a kitchen stove

Goal: Answer the query "What kind of appliance is the bowl sitting on top of?"
# Think: Can i get the answer directyly from the previous information?
# Think: No.But the caption implies that the scene in the picture is a kitchen, so the appliances should also be common kitchen appliances

# Think:In order to find the appliance under the bowl, I first needed to find the bowl
Step 1:{"role": "finder", "input_variable": "image", "task_description": "find the bowl ", "expected_output_result": {"bowl_patch": "The bowl in the image"}}
# Think:Next i should find the appliance under the bowl
Step 2:{"role": "finder", "input_variable": "bowl_patch", "task_description": "find the kitchen appliance under the bowl ", "expected_output_result": {"appliance_patch": "The appliance under the bowl"}}
# Think:I'll let querier tell me what kind of appliance it is.
Step 3:{"role": "querier", "input_variable": "appliance_patch", "task_description": "What kind of appliance is this?", "expected_output_result": {"appliance_type": "The type of appliance"}}
Step 4:{"role": "OUTPUT", "input_variable": "appliance_type", "output_variable": "appliance_type"}
(Example 7 finished)
(Example 8 start):
Information from previous run: 
- ImageCaption : a table with a vase of flowers on it
Goal: Answer the query "On which side is the bulb?"
# Think: Can i get the answer directyly from the previous information?
# Think: No.Because it does not mention the side of the bulb.I need to make a plan.

# Think:At first i should find the bulb
Step 1:{"role": "finder", "input_variable": "image", "task_description": "find the bulb ", "expected_output_result": {"bulb_patch": "The bulb in the image"}}
# Think:Next i should get the bulb position
Step 2:{"role": "querier", "input_variable": "bulb_patch", "task_description": "On which side is the bulb?", "expected_output_result": {"bulb_side": "The side of the bulb in this image"}}
Step 3:{"role": "OUTPUT", "input_variable": "bulb_side", "output_variable": "The bulb is on the {bulb_side} side of the vase."}
(Example 8 finished)
'''

plan_prompt_V4=f'''
You are autogpt, and you are working with multiple AIs of different abilities to solve a problem about an image,and you're the planner.
First you get information about the image as a whole provided by the captioner,then write an abstract plan to successfully answer the query. 
In each step of the plan you need to provide the AI to be called, input parameters, and expected results in the form of json. Learn from and incorporate information from previous runs.

You can call the following AIs:\n
{"".join(AI_des_V2)}

Here are examples:
{Examples_V4}
(All Examples finished)

Here is a new query.First you get information about the image as a whole provided by the captioner,then write an abstract plan to successfully answer the query. 
In each step of the plan you need to provide the AI to be called, input parameters, and expected results in the form of json. Learn from and incorporate information from previous runs.
Remeber:
1.Follow the format of the python code when using the output function at the end.
2.Try to infer more information from the image caption first in your thinking process, and then make a plan

Information from previous run: {{(previous_information)}}
Goal: Answer the query "{{(query)}}"

'''

class OutPuter:

    def __init__(self) -> None:
        self.conversation_history = []
    #Output must need an input variable
    def parse(self,inputs:dict):
        input_variable = inputs.get('usable')
        codeline = inputs.get('varname')
        if len(input_variable)>0:
            for variable in input_variable.split(';'):
                    if len(variable)>0:
                        v_name = variable.split(':')
                        if len(v_name)>1:
                            v_name,des = v_name
                        else:
                            v_name = v_name[0]
                        v_name = v_name.strip()
                        v_2 = '{'+v_name+'}'
                        if v_2 not in codeline:
                            codeline = codeline.replace(v_name,v_2)
        result = f"f'{codeline}'"
        return [result]

    
    def restart(self):
        self.conversation_history = []
outputer = OutPuter()