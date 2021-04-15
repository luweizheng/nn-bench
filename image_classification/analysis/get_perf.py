import os
import math
import numpy as np
import json
import sys

# the first argument is the deep learning framwork: pytorch
# the second argument is the platform: gpu/npu
if len(sys.argv) > 2:
    framework = sys.argv[1]
    platform = sys.argv[2]
    path = "../" + framework + "/output/" + platform
    if not os.path.isdir(path):
        print('Error: ', path, ' is not a directory.')
        exit()
else:
    print('Please specify output data directory.')
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
            if 'time:' in line:
                time = float(line[len('time:') + 1:])
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

with open('./data/' + framework + '_' + platform + '.json', 'w') as outputfile:
    json.dump(data, outputfile)