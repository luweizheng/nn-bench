import os
import math
import numpy as np
import json
import sys

root = '../output/'

# the first argument is the subfolder of `output`
if len(sys.argv) > 1:
    path = root + sys.argv[1]
    if not os.path.isdir(path):
        print('Error: ', path, ' is not a directory.')
        choices = os.listdir(root)
        if len(choices) == 1:
            print('There is only one directory. Running', choices[0])
            path = root + choices[0]
        else:
            print('Available directories: ', os.listdir(root))
            exit()
else:
    choices = os.listdir(root)
    if len(choices) == 1:
        print('There is only one directory. Running', choices[0])
        path = root + choices[0]
    else:
        print('Please the input directory.')
        print('Available directories: ', os.listdir(root))
        exit()

files = os.listdir(path)

state_list = []
time_list = []
flops_list = []
memory_list = []
eps_list = []
ai_list = []
params_list = []

for filename in files:
    '''
    '''
    with open(path+'/'+filename) as f:
        lines = f.readlines()
        state = "fine"
        time, flops, memory, example_per_sec, arithemetic_intensity, params = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        for line in lines:
            if 'Traceback' in line:
                state = "error"
            if 'device time:' in line:
                time = float(line[len('device time:') + 1:])
            if 'flops:' in line:
                flops = float(line[len('flops:') + 1:])
            if 'memory:' in line:
                memory = float(line[len('memory:') + 1:])
            if 'example_per_sec:' in line:
                example_per_sec = float(line[len('example_per_sec:') + 1:])
            if 'parameter size:' in line:
                params = int(line[len("parameter size:") + 1:])
            if 'arithemetic intensity:' in line:
                arithemetic_intensity = float(line[len('arithemetic intensity:') + 1:])
        
        if time == 0.0:
            state = "error"

        state_list.append(state)
        time_list.append(time)
        eps_list.append(example_per_sec)
        flops_list.append(flops)
        memory_list.append(memory)
        ai_list.append(arithemetic_intensity)
        params_list.append(params)

data = {
    'labels': files,
    'state': state_list,
    'device_time': time_list,
    'flops': flops_list,
    'memory': memory_list,
    'example_per_sec': eps_list,
    'arithemetic_intensity': ai_list,
    'params': params_list
}

with open('./data/' + sys.argv[1] + '.json', 'w') as outputfile:
    json.dump(data, outputfile)
