from pygments.lexers.c_cpp import CLexer
import os
import json
import gzip
import pickle
from tqdm import tqdm
import multiprocessing
import random
import glob
import multiprocessing
import itertools
import pickle
import time
import random
import codecs
# import openai
random.seed(2024)
# openai.api_key = "sk-XXXX"
from openai import OpenAI
client = OpenAI(api_key="sk-XXXXX", base_url="https://api.deepseek.com")

system_prompt = '''
You are an advanced code summarization tool. 
Given the source code of a function, generate a summary for the function.
Provide response in the following format: input:<INPUTS> | output:<OUTPUTS> | behavior:<BEHAVIOR1,>, <BEHAVIOR2>, ... 
where <INPUTS> lists the parameters the function accepts, <OUTPUTS> lists the return values of the function (not the type), 
and <BEHAVIOR> concisely describes the behavior of the function in conjunction with the variables of input and output, 
without any other details other than the operations performed.
'''

def user_prompt_basic(source_code):
	user_prompt = 'Only provide the response in the format mentioned before for the following code snippet: \n' + \
				   source_code
	return user_prompt


def user_prompt_behavior_guide(source_code):
	user_prompt =  '''The behaviors should fall into the following behavior types:
						1. Access Control: Describe any actions related to managing permissions.
						2. Arithmetic Operations: Describe any arithmetic calculations performed.
						3. Authentication: Describe any actions related to verifying identities.
						4. Code Execution: Describe any actions related to executing code or commands.
						5. Command Injection: Describe any actions related to handling command injection risks.
						6. Concurrency: Describe any actions related to managing multiple threads or processes.
						7. Data Serialization: Describe any actions related to converting data formats.
						8. File Handling: Describe any actions related to reading or writing files.
						9. Input Validation: Describe any actions related to checking or sanitizing inputs.
						10.Memory Management: Describe any actions related to allocating or freeing memory.
						11.Network Communication: Describe any actions related to sending or receiving data over a network.
						12.Path Management: Describe any actions related to managing file or directory paths.
						13.Session Management: Describe any actions related to handling user sessions or states.
					Only describes in detail the behavior of the function along with the input and output variables, without specifying the behavior types. 
					''' 

	user_prompt = user_prompt + "\n" + 'Only provide the response in the format mentioned before for the following code snippet: \n' + \
				  source_code
	return user_prompt

def user_prompt_example(source_code):
	user_prompt =  '''Given an example as follow:
					  char* concatenate(const char *str1, const char *str2) {
						char buffer[BUFFER_SIZE];
						strcpy(buffer, str1);
						strcat(buffer, str2); 
						char *result = (char *)malloc(strlen(buffer) + 1);
						if (result == NULL) {
							printf("Memory allocation failed\n");
							exit(1);
						}
						strcpy(result, buffer);
						return result;
					}

					The summary is :input:const char *str1, const char *str2 | output:char* result | behavior:Copy str1 to buffer, Append str2 to buffer, Allocate memory for result, Copy buffer to result, Handle memory allocation failure.
					''' 
	user_prompt = user_prompt + "\n" + 'Only provide the response in the format mentioned before for the following code snippet: \n' + \
				  source_code
	return user_prompt

def user_prompt_cot(source_code):
	user_prompt =  "Let us think step by step. Only provide the response in the format mentioned before for the following code snippet: \n" + \
				  source_code
	return user_prompt

def code_retrieve(question):
	completion = openai.ChatCompletion.create(
		model="gpt-4",
		temperature=1,
		max_tokens=100,
		messages=[
			{"role": "user", "content": question}
		]
	)
	message = completion.choices[0].message.content
	return message

def code_summarization(code, ptid):
	if ptid == 0:
		user_content = user_prompt_basic(code)
	elif ptid == 1:
		user_content = user_prompt_behavior_guide(code)
	elif ptid == 2:
		user_content = user_prompt_example(code)
	elif ptid == 3:
		user_content = user_prompt_cot(code)
	else:
		user_content = user_prompt_basic(code)

	user_content = user_content[:7000]
	# print(user_content)
	# completion = openai.ChatCompletion.create(
	# 	model="davinci-002",
	# 	temperature=1,
	# 	max_tokens=500,
	# 	messages=[
	# 		{"role": "system", "content": system_prompt},
	# 		{"role": "user", "content": user_content}
	# 	]
	# )
	# message = completion.choices[0].message.content
	response = client.chat.completions.create(
		model="deepseek-coder",
		messages=[
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_content},
		],
		stream=False
	)
	message = response.choices[0].message.content

	# time.sleep(0.5)
	return message


def generate_summary_for_callees(callees, ptid):
	summaries = []
	last_layer = callees[-1]['layer']
	for i in tqdm(range(last_layer, 0, -1)):
		if i == last_layer:
			for callee in callees:
				if callee['layer'] == i:
					summary = code_summarization(callee['func_str'], ptid)
					print(summary)
					summaries.append({'layer':callee['layer'], 'func_name':callee['func_name'], 'summary':summary, 'caller':callee['caller']})
		else:
			for callee in callees:
				if callee['layer'] == i:
					func_str = callee['func_str']
					func_name = callee['func_name']
					for summary in summaries:
						if summary['caller'] == func_name:
							func_str = func_str + summary['summary']
					summary = code_summarization(func_str, ptid)
					print(summary)
					summaries.append({'layer':callee['layer'], 'func_name':callee['func_name'], 'summary':summary, 'caller':callee['caller']})
	return summaries


def generate_summary(inpath, out, dataset, ptid):
	with open(os.path.join(out,'enhance', dataset, str(ptid),'devign_summary.jsonl'), "a+") as f:	
		with open(inpath, 'r') as ff:
			items = list(ff)
			for item in items:
				instance = {}
				content = json.loads(item)
				callees = content['callee']
				summaries = generate_summary_for_callees(callees, ptid)
				instance['idx'] = content['idx']
				instance['target'] = content['target']
				instance['project'] = content['project']
				instance['commit_id'] = content['commit_id']
				instance['file'] = content['file']
				instance['func'] = content['func']
				instance['func_name'] = content['func_name']
				instance['callee'] = content['callee']
				instance['summary'] = summaries
				f.write(json.dumps(instance)+'\n')
				
