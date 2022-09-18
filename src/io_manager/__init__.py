import json
import pickle

import yaml
import os


def load_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def write_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_json(path):
    with open(path, 'r') as f:
        obj = json.load(f)
    return obj


def write_list(list_obj, path):
    with open(path, 'w') as f:
        f.write(','.join(list_obj))
    print(f"Columns written to {path}")


def load_list(path):
    print(f"Reading columns from {path}")
    with open(path, 'r') as f:
        return f.read().split(',')


def read_yaml():
    print(__file__)
    print(os.listdir(".."))
    with open("./config.yml", 'r') as f:
        ret = yaml.safe_load(f)
    return ret
