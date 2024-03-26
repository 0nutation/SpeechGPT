import json
import glob
from random import shuffle


with open('librilight_discretize_audio_pretrain_2k.json', 'r') as fp:
    data = json.load(fp)


result = list()
for sample in data:
    result.append(sample['conversations'][-1]['value'] + '\r\n')


shuffle(result)
with open('train.txt', 'w') as fp:
    fp.writelines(result[:int(len(result) * 0.8)])
with open('dev.txt' , 'w') as fp:
    fp.writelines(result[int(len(result) * 0.8):])
