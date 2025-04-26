from modulefinder import STORE_GLOBAL
from tqdm import tqdm
import json
import os
import argparse
import random
import librosa
from sklearn.model_selection import train_test_split
import subprocess

cnt = 0

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def json_dump(filename, ob):
    with open(filename, 'w', encoding = 'utf-8') as file:
        json.dump(ob, file, ensure_ascii = False, indent = 4)  

def huhu(j):
     global cnt
     j_new = {}
     for i in j:
        if 'textlast' in i:
             i['text'] = i['textlast']
             del i['textlast']
        j_new[f'ex{cnt}'] = i
        cnt += 1
     return j_new

def main():
    parser = argparse.ArgumentParser(description = 'Set up json files')
    parser.add_argument("--json", "-j", type = str, default = r"dsp_new.json", help = "json path")
    parser.add_argument("-number", "-n", type = float, default = 1, help = "Number of files")
    parser.add_argument("--seed", "-s", type = int, default = 123456, help = "Random Seed")
    parser.add_argument("output", type = str, default = "dsp_train", help = "output name")
    args = parser.parse_args()
    assert args.json
    with open(args.json, 'r', encoding = 'utf-8') as file:
        a = file.read()
        j_list = json.loads(a)

    j_list = [x for x in j_list if (not has_numbers(x['textlast']))]
    j_list = random.Random(args.seed).sample(j_list, int(args.number*len(j_list)))

    j_train, j_test = train_test_split(j_list, train_size = 0.8, random_state = args.seed)
    j_valid, j_test = train_test_split(j_test, train_size = 0.5, random_state = args.seed)

    j_train = huhu(j_train)
    j_valid = huhu(j_valid)
    j_test = huhu(j_test)

    json_dump(f"{args.output}_train.json", j_train)
    json_dump(f"{args.output}_valid.json", j_valid)
    json_dump(f"{args.output}_test.json", j_test)


if __name__ == '__main__':
	main()
