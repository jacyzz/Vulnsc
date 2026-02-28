# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import logging
import sys
import json
import numpy as np
import json
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
import re
import csv


def evaluate_predictions(labels, predicted_prob, files, args):
    predicted_labels = [0 if i <0.5 else 1 for i in predicted_prob]
    acc = accuracy_score(labels, predicted_labels)
    reports = classification_report(labels, predicted_labels, target_names=['0', '1'], output_dict=True)
    neg_pre = reports['0']['precision']
    neg_rec = reports['0']['recall']
    neg_f1 = reports['0']['f1-score']
    pos_pre = reports['1']['precision']
    pos_rec = reports['1']['recall']
    pos_f1 = reports['1']['f1-score']
    prc_auc = average_precision_score(labels, predicted_prob)
    metrics = {'Acc': acc, 'Pos_pre': pos_pre, 'Pos_rec': pos_rec, 'Pos_f1': pos_f1}
    with open("../cb_result.csv","a+") as f:
        writer = csv.writer(f)
        writer.writerow([args.model, args.enhance, args.prompt, args.seed])
        writer.writerow(metrics.keys())
        writer.writerow(metrics.values())
    # evaluate_cal(predicted_prob, labels, files)
    return metrics


def read_answers(filename):
    answers={}
    files = {}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            js=json.loads(line)
            answers[js['idx']]=js['target']
            files[js['idx']] = js['file']
    return answers, files

def read_predictions(filename):
    predictions={}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            idx,label=line.split()
            predictions[int(idx)]=float(label)
    return predictions


def calculate_scores(answers,predictions,files,args):
    labels = []
    predicted_labels = []
    predicted_files = []
    for k, v in answers.items():
        labels.append(v)
        if k not in predictions.keys():
            logging.error("Missing prediction for index {}.".format(k))
            sys.exit()
        predicted_labels.append(predictions[k])
        predicted_files.append(files[k])

    metrics = evaluate_predictions(labels,predicted_labels,predicted_files,args)
    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for Defect Detection dataset.')
    parser.add_argument('--answers', '-a',help="filename of the labels, in json format.")
    parser.add_argument('--predictions', '-p',help="filename of the leaderboard predictions, in txt format.")
    parser.add_argument('--model', '-m',help="model name.")
    parser.add_argument('--enhance', '-e',help="enhance or not.")
    parser.add_argument('--prompt', '-t',help="prompt name.")
    parser.add_argument('--seed', '-s',help="seed name.")
    

    args = parser.parse_args()
    answers, files = read_answers(args.answers)
    predictions=read_predictions(args.predictions)
    scores=calculate_scores(answers,predictions,files,args)
    print(scores)

if __name__ == '__main__':
    main()
