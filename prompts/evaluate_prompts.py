evaluate_prompt_v1='''
You are the examiner of the test and you are responsible for judging whether the student's answer matches the correct answer and outputting

Query:Are there any women?
CorrectAnswer:No, there are no women.
StudentAnswer:No
Judge:Correct.Student answer matches the correct answer

Query:On which side of the picture is the clock?
CorrectAnswer:The clock is on the right of the image.	
StudentAnswer:right	
Judge:Correct.Student answer matches the correct answer

Query:What is this furniture?
CorrectAnswer:table
StudentAnswer:desk	
Judge:Correct.Student answer is semantically identical to the correct answer

Query:The lid on top of the garbage bin is what color?
CorrectAnswer:The lid is green.
StudentAnswer:orange
Judge:Wrong.The student answer is completely different from the correct answer

Query:What is the boy holding?
CorrectAnswer:The boy is holding the mobile phone.
StudentAnswer:It depends on the context.
Judge:Wrong.Student's answer is irrelevant to the question.

Query:What device on the desk?
CorrectAnswer:The device is a laptop.
StudentAnswer:No device found to the left of the cream.
Judge:Wrong.Student's answer is irrelevant to the question.

Query:What animals behinde the fence?
CorrectAnswer:It is a bear.
StudentAnswer:bears.
Judge:Correct.The student's answer is the same as the correct answer to the question.

Query:What furniture under the microwave?
CorrectAnswer:The furniture is a table.
StudentAnswer:Error in execution of statement "ANSWER = wooden_furniture[0].simple_query("What kind of furniture is this?")":list index out of range
Judge:Wrong.The student's answer is completely irrelevant to the question

Query:What is the drawer underneath?
CorrectAnswer:The drawers are beneath the counter.
StudentAnswer:counter
Judge:Correct.The student's answer is the same as the correct answer to the question.

Query:{query}
CorrectAnswer:{answer}
StudentAnswer:{model_output}
Judge:
'''

evaluate_prompt_v2='''
You are an examiner who can judge whether a student's answer matches the correct answers. 
Next, I will provide you with the correct answer and a student's answer. Please judge whether the student's answer matches the correct answers.
Query:What furniture under the microwave?
CorrectAnswer:The furniture is a table.
StudentAnswer:Error in execution of statement "ANSWER = wooden_furniture[0].simple_query("What kind of furniture is this?")":list index out of range
Judge:Wrong.The student's answer is completely irrelevant to the question

Query:What is the drawer underneath?
CorrectAnswer:The drawers are beneath the counter.
StudentAnswer:counter
Judge:Correct.The student's answer is the same as the correct answer to the question.

Query:{query}
CorrectAnswer:{answer}
StudentAnswer:{model_output}
Judge:
'''