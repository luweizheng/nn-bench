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

dtype_list=[]
batch_list=[]
input_list=[]
output_list=[]
time_list=[]
flops_list=[]
memory_list=[]
eps_list=[]
ai_list=[]
fl_list=[]

for filename in files:
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
    with open(path+'/'+filename) as f:
        lines=f.readlines()
        for line in lines:
            if 'device time:' in line:
                time=float(line[13:])
            if 'flops:' in line:
                flops=float(line[7:])
            if 'memory:' in line:
                memory=float(line[8:])
            if 'example_per_sec:' in line:
                example_per_sec=float(line[17:])
            if 'arithemetic intensity:' in line:
                arithemetic_intensity=float(line[22:])
        time_list.append(time)
        eps_list.append(example_per_sec)
        flops_list.append(flops)
        memory_list.append(memory)
        ai_list.append(arithemetic_intensity)

data={
    'labels':files,
    'dtype':dtype_list,
    'batch_size':batch_list,
    'input_size':input_list,
    'output_size':output_list,
    'device_time':time_list,
    'flops':flops_list,
    'memory':memory_list,
    'example_per_sec': eps_list,
    'arithemetic_intensity':ai_list
}

#print(data)

with open(sys.argv[1] + '.json', 'w') as outputfile:
    json.dump(data, outputfile)