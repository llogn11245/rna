import json

import os

def load_json(path):
    """
    Load a json file and return the content as a dictionary.
    """
    with open(path, "r", encoding = 'utf-8') as f:
        data = json.load(f)
    return data

def save_data(data, data_path):
    with open(data_path, "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def check_and_save(path):
    res = []
    data = load_json(path)
    for item in data:
        if os.path.exists(item["wav_path"]):
            print(f"File {item['wav_path']} found!")
            res.append(item)
    save_data(res, path)

check_and_save("/home/anhkhoa/transformer_transducer/data/train.json")
check_and_save("/home/anhkhoa/transformer_transducer/data/dev.json")
check_and_save("/home/anhkhoa/transformer_transducer/data/test.json")