import json
from collections import namedtuple

default_dict = {}
with open('options/default.json') as f:
    default_dict = json.load(f)

default = namedtuple("default", default_dict.keys())(*default_dict.values())
