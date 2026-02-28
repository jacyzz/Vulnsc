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
from scripts.code_summary import generate_summary
from scripts.code_retrieval import code_retrieve
from scripts.code_enhance import instantiate_summary, split_enhance_dataset

if __name__ == '__main__':
	## for devign 
	## code retrieval
	n_layer = 6  ## maximum layers
	outpath = os.path.join(os.getcwd(), 'data')
	inpath = os.path.join(os.getcwd(), 'data', 'origin', 'devign.json')
	
	repo_path = os.path.join(os.getcwd(), 'data', 'repo', 'qemu')
	if not os.path.exists(repo_path):
		subprocess.run(['git', 'clone', 'https://github.com/qemu/qemu.git', str(repo_path)])
	repo_path = os.path.join(os.getcwd(), 'data', 'repo', 'FFmpeg')
	if not os.path.exists(repo_path):
		subprocess.run(['git', 'clone', 'https://github.com/FFmpeg/FFmpeg.git', str(repo_path)])

	code_retrieve(inpath, outpath, n_layer)

	## code summary
	in_file = os.path.join(outpath, 'enhance',  'devign_source.jsonl')
	for i in range(0,4):
		prompt_id = i
		generate_summary(in_file, outpath, 'devign', prompt_id)

	## code enhance and split
	out = os.path.join(outpath, 'enhance', 'devign')
	for i in range(0,4):
		ptid = i
		inpath = os.path.join(out, str(ptid), "devign_summary.jsonl")
		instantiate_summary(inpath, out, ptid)
		split_enhance_dataset(out, ptid)

