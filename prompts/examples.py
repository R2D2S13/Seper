Example_V1='''
Example 1:
Query: Is the plate the same color as the curtain?
function_call: find_objects('image','plate')
function: There are 3 plates in the image.USABLE_IMAGE_PATCH:plates_1,plates_2,plates_3
function_call: find_objects('image','curtain')
function: There are 1 curtain in the image.USABLE_IMAGE_PATCH:plates_1,plates_2,plates_3,curtain_1
function_call: simple_query('plates_1','What is the color?')
function: The answer of 'What is the color?' in 'plates_1' patch is red.
function_call: simple_query('curtain_1','What is the color?')
function: The answer of 'What is the color?' in 'curtain_1' patch is red.
function_call: finish('give_answer',final_answer='Yes, both the plate and the curtain are red.')

Example 2:
Query: Is the brown animal to the right or to the left of the young man?
function_call: find_objects('image','brown animal')
function: There are 1 brown animal in the image.USABLE_IMAGE_PATCH:brown animal_1
function_call: find_objects('image','man')
function: There are 1 man in the image.USABLE_IMAGE_PATCH:brown animal_1,man_1
function_call: pos_relation('brown animal_1','man_1')
function: The brown animal_1 is on the left of the man_1
function_call: finish('give_answer',final_answer='The brown animal is to the left of the young man.')

Example 3:
Query: What is the shape of the bar furthest away?
function_call: find_objects('image','bar')
function: There are 2 bars in the image.USABLE_IMAGE_PATCH:bar_1,bar_2
function_call: compute_depth('bar_1')
function: The depth of bar_1 is 35
function_call: compute_depth('bar_2')
function: The depth of bar_2 is 48
function_call: simple_query('bar_2','What is the shape?')
function: The answer of 'What is the shape?' in 'bar_2' patch is triangle.
function_call: finish('give_answer',final_answer='triangle')

'''