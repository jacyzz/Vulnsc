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
import random
from tqdm import tqdm
import codecs
import subprocess
import re


def generate_cscope_database(source_dir):
	"""
	Generate cscope database for the given source directory.
	"""
	# Find all folders in source_dir into cscope.files
	# subprocess.run(['find', '.', '-name', '"*.c"', '>', 'cscope.files'])
	result = subprocess.run(['find', '.', '-name', '*.c'], stdout=subprocess.PIPE, text=True)
	with open('cscope.files', 'w') as file:
		file.write(result.stdout)

	# Generate the cscope database
	subprocess.run(['cscope', '-b', '-q', '-k', '-i', 'cscope.files'])

def get_callees(source_dir, function_name):
	"""
	Get the called functions of funcname.
	"""
	# Get the called functions of function_name
	result = subprocess.run(['cscope', '-dL2' + function_name], stdout=subprocess.PIPE, text=True)
	lines = result.stdout.splitlines()
	if len(lines) == 0:
		return "", None

	called_functions = []
	file_path = lines[0].split()[0]
	for line in lines:
		parts = line.split()
		if len(parts) > 1 and parts[0] == file_path and parts[1] not in called_functions :
			called_functions.append(parts[1])
		# print(line)
	return file_path, called_functions

def extract_func_name(file_path):
	"""
	Extract functions according to function name in file_path.
	"""	

	# Extract source code of function_name
	function_name = ""

	result = subprocess.run(['ctags', '--fields=+ne-t', '-o', '-', '--sort=no', '--excmd=number', str(file_path)], stdout=subprocess.PIPE, text=True)
	lines = result.stdout.splitlines()
	if len(lines) == 0:
		return function_name
	fields = lines[0].split()
	function_name = fields[0]
	return function_name


def extract_func(source_dir, file_path, function_name):
	"""
	Extract functions according to function name in file_path.
	"""	
	# Extract source code of function_name
	result = subprocess.run(['ctags', '--fields=+ne-t', '-o', '-', '--sort=no', '--excmd=number', str(file_path)], stdout=subprocess.PIPE, text=True)
	lines = result.stdout.splitlines()
	for line in lines:
		fields = line.split()
		if 'f' in fields and fields[0] == function_name:
			start_num, end_num = extract_numbers(line)
			if start_num == None:
				return None
			try:
				func_str = extract_func_str(file_path, start_num, end_num)
				return func_str
			except:
				return None				
	return None

def extract_numbers(text):
	"""
	Extract numbers following 'line:' and 'end:' from the given text.
	
	Parameters:
	text (str): The input text containing 'line:' and 'end:'.
	
	Returns:
	tuple: A tuple containing (line_number, end_number), or (None, None) if not found.
	"""
	# Regular expression to match 'line:' and 'end:' followed by numbers
	pattern = r'line:(\d+).*end:(\d+)'
	
	match = re.search(pattern, text)
	
	if match:
		line_number = int(match.group(1))
		end_number = int(match.group(2))
		return line_number, end_number
	else:
		return None, None

def extract_func_str(file_path, start_num, end_num):
	with open(file_path, "r", encoding='utf-8') as rfile:
		lines = rfile.readlines()
		return "".join(lines[start_num - 1:end_num])


def retrieve_callee(source_dir, commit_id, function_name, n_layer=None):
	callees = []
	# Change to the source directory
	os.chdir(source_dir)

	# Git checkout to given commit version
	subprocess.run(['git', 'checkout', '-f', str(commit_id)])

	# Generate cscope database
	generate_cscope_database(source_dir)

	# Get file path and callee of function_name
	layer = 1
	funcs_temp = []
	cfs = []
	file_path, called_functions = get_callees(source_dir, function_name)
	if file_path == "":
		print("No callees in root function.")
		return callees
	for called_function in called_functions:
		funcs_temp.append({'layer':layer,'func_name':called_function,'caller':function_name})
		cfs.append(called_function)

	# print(funcs_temp)

	while True:
		temp = []
		layer = layer + 1
		for func in funcs_temp:
			funcs_temp.remove(func)
			file_path, called_functions = get_callees(source_dir, func['func_name'])
			if file_path == "":
				# print("cscope no file path.")
				continue
			func_str = extract_func(source_dir, file_path, func['func_name'])
			if func_str == None:
				# print("ctages no function code.")
				continue
			callee = {'layer':func['layer'], 'func_name':func['func_name'], 'func_str':func_str, 'caller':func['caller']}
			callees.append(callee)
			for called_function in called_functions:
				if called_function not in cfs:
					cfs.append(called_function)
					temp.append({'layer':layer,'func_name':called_function,'caller':function_name})
		funcs_temp.extend(temp)
		if len(funcs_temp) == 0:
			break
		if layer == n_layer:
			break

	return callees


def code_retrieve(file_path, out, n_layer):
	instances = []
	with open(file_path, 'r') as f:
		content = f.read()
		instances = json.loads(content)

	datas = []
	for i, item in tqdm(enumerate(instances), total=len(instances)):
		data = {}
		data['idx'] = i
		data['target'] = item['target']
		data['project'] = item['project']
		data['commit_id'] = item['commit_id']
		data['file'] = str(item['target'])+"_"+str(item['project']) +"_"+str(item['commit_id'])+".c"
		data['func'] = item['func']

		filepath = os.path.join(out,"temp",data['file'])
		with open(filepath,"w") as file:
			file.write(item['func'])
		function_name = extract_func_name(filepath)
		if function_name == "":
			continue
		data['func_name'] = function_name
		datas.append(data)

	with open(os.path.join(out,'enhance','devign_source.jsonl'), "w") as f:
		for data in tqdm(datas):
			if data['project'] == "qemu":
				source_dir = os.path.join(out, 'repo', 'qemu')
			if data['project'] == "FFmpeg":
				source_dir = os.path.join(out, 'repo', 'FFmpeg')
			callees = retrieve_callee(source_dir, data['commit_id'], data['func_name'], n_layer)
			if len(callees) == 0 :
				continue
			data['callee'] = callees
			f.write(json.dumps(data)+'\n')

