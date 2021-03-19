import os
import math
import numpy as np
import json
import sys

root = '../output/linear/'

files = os.listdir(root)

dtype_list=[]
batch_list=[]
input_list=[]
output_list=[]
time_list=[]
flops_list=[]
memory_list=[]
ai_list=[]
fl_list=[]

for filename in files:
    #print(filename)
    splitname=filename.split('-')
    #print(splitname)
    '''
    splitname[0]: platform
    splitname[1]: compute type
    splitname[2]: dtype
    splitname[3]: batch size
    splitname[5]: input size
    splitname[7]: output size
    '''
    dtype_list.append(splitname[2])
    tempbatch=splitname[3].split('_')
    batch_list.append(int(tempbatch[1]))
    input_list.append(int(splitname[5]))
    output_list.append(int(splitname[7]))
    with open(root+filename) as f:
        contents=f.readlines()
        time=float(contents[4][13:])
        flops=float(contents[5][7:])
        memory=float(contents[6][8:])
        arithemetic_intensity=float(contents[7][22:])
        flops_scale=(contents[8][14:])
        if(int(tempbatch[1])==512):
            print(flops_scale)
        time_list.append(time)
        flops_list.append(flops)
        memory_list.append(memory)
        ai_list.append(arithemetic_intensity)
        fl_list.append(flops_scale)

data={
    'labels':files,
    'dtype':dtype_list,
    'batch_size':batch_list,
    'input_size':input_list,
    'output_size':output_list,
    'device_time':time_list,
    'flops':flops_list,
    'memory':memory_list,
    'arithemetic_intensity':ai_list,
    'flops_scale':fl_list
}

#print(data)

with open("npu_linear.json",'w') as outputfile:
    json.dump(data, outputfile)