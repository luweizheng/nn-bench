import os
import json

'''
Read a json file, return a dict.
'''
def get_data(filename = ''):
    f = './data/' + filename + '.json'
    if not os.path.isfile(f):
        print(filename + '.json', 'does not exist.')
        return None
    with open(f, 'r') as infile:
        d = json.load(infile)
    return d