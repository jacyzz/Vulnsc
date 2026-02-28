from pygments.lexers.c_cpp import CLexer
import os
import json
import gzip
import pickle
from tqdm import tqdm
import multiprocessing
import random
import glob
from sklearn.model_selection import train_test_split
import multiprocessing
import itertools
import pickle
import random
import codecs
import subprocess
import re

def extract_function_parameters(c_code, function_name):
	pattern = rf'\b{function_name}\s*\(([^)]*)\)'
	matches = re.findall(pattern, c_code)
	function_params = []
	
	for params in matches:
		# print(params)
		params_list = [param.strip() for param in params.split(',') if param.strip()]
		function_params.append({'func_statement':function_name+"("+str(params)+")" ,'params': params_list})

	return function_params


def instantiate_summary(inpath, out, llm, n, ptid):
	with open(os.path.join(out,llm,str(ptid),'devign_enhance_'+str(n)+'.jsonl'), "w") as f:
		with open(inpath, 'r') as ff:
			items = list(ff)
			for item in items:
				instance = {}
				content = json.loads(item)
				if not isinstance(content, dict):
					print(i, content)
					continue
				instance['idx'] = content['idx']
				instance['target'] = content['target']
				instance['project'] = content['project']
				instance['commit_id'] = content['commit_id']
				instance['file'] = content['file']
				instance['func'] = content['func']
				func_str = content['func']
				for summary in content['summary']:
					if summary['layer'] == 1:
						func_name = summary['func_name']
						summary_str = summary['summary'].replace("\n","").replace("\r","")
						summary_str = re.sub(r'\s+', ' ', summary_str)
						summary_str = summary_str.strip()
						func_str = func_str + "; " +summary_str
				instance['func_en'] = func_str
				f.write(json.dumps(instance)+'\n')


def split_enhance_dataset(out, ptid):
	samples = []
	with open(os.path.join(out,str(ptid),'devign_enhance.jsonl'), "r") as f:	
		items = list(f)
		for item in tqdm(items):
			samples.append(json.loads(item))
	
	nonvul = [i for i in samples if i["target"]==0]
	vul = [i for i in samples if i["target"]==1]
	print("samples:"+str(len(samples)))
	print("vul:"+str(len(vul))+", nonvul:"+str(len(nonvul)))
	
	### split the dataset 
	random.shuffle(vul)
	random.shuffle(nonvul)
	train_dataset, valid_dataset, test_dataset = [], [], []
	train_nonvul_dataset, test_nonvul_dataset = train_test_split(nonvul, test_size=0.2, random_state=2025)
	train_vul_dataset, test_vul_dataset = train_test_split(vul, test_size=0.2, random_state=2025)
	test_nonvul_dataset, valid_nonvul_dataset = train_test_split(test_nonvul_dataset, test_size=0.5, random_state=2025)
	test_vul_dataset, valid_vul_dataset = train_test_split(test_vul_dataset, test_size=0.5, random_state=2025)
	train_dataset.extend(train_nonvul_dataset)
	train_dataset.extend(train_vul_dataset)
	valid_dataset.extend(valid_nonvul_dataset)
	valid_dataset.extend(valid_vul_dataset)
	test_dataset.extend(test_nonvul_dataset)
	test_dataset.extend(test_vul_dataset)
	
	with open(os.path.join(out,str(ptid),'train.jsonl'), "w") as f:
		for sample in train_dataset:
			f.write(json.dumps(sample)+'\n')

	with open(os.path.join(out,str(ptid),'valid.jsonl'), "w") as f:
		for sample in valid_dataset:
			f.write(json.dumps(sample)+'\n')

	with open(os.path.join(out,str(ptid),'test.jsonl'), "w") as f:
		for sample in test_dataset:
			f.write(json.dumps(sample)+'\n')

	print("train:"+str(len(train_dataset))+", valid:"+str(len(valid_dataset))+", test:"+str(len(test_dataset)))

