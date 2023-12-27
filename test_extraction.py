from dataset_extraction import read_canbus, two_byte, one_byte
import json
import os
from collections import namedtuple, defaultdict
from functools import partial
from itertools import zip_longest
from datetime import datetime, timedelta
import traceback
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

bom_basinc_obj = namedtuple('BOM_BASINC', ('time', 'value'))
kepce_basinc_obj = namedtuple('KEPCE_BASINC', ('time', 'value'))

bom_aci_obj = namedtuple('BOM_ACI', ('time', 'value'))
kepce_aci_obj = namedtuple('KEPCE_ACI', ('time', 'value'))
valve_obj = namedtuple('VALVE', ('time', 'x_axis', 'y_axis', 'kepce_acma', 'kepce_kapama', 'bom_kald', 'bom_ind'))
pedal_volt_obj = namedtuple('VOLT', ('time', 'value'))

significant_types = {'1E5': 'kepce_aci',
                     '1E7': 'bom_aci',
                     'CF00400x': 'devir',
                     '10D': 'joystick',
                     '152': 'vites',
                     '18FF1E03x': 'hiz',
                     '153': 'lamba',
                     'CF00300x' : 'pedal_volt',
                     '18FF1E03x' : 'km_sa',
                     '151' : 'fuel_level'}


df = pd.DataFrame(np.nan, index=range(47), columns=['bom_basinc', 'kepce_basinc','bom_aci', 'kepce_aci', 'valve_x', 'valve_y', 'pedal_volt'] )

def read_bom_kepce():
    segment_kepce =[[] for i in range(47)]
    segment_bom =[[] for i in range(47)]
    i = 0
    while i < 47:
        path = os.path.join('test_dataset', 'datalogger_10.23', f"{i+1}_221023.txt") #test_dataset\datalogger_10.23\1_221023.txt
        with open(path, 'r') as fh:
            text = fh.read()

        lines = text.splitlines()
        bom = list(map(float, lines[22].split()[1:-1]))
        kepce = list(map(float, lines[25].split()[1:-1]))
        start = lines[-3].split()[-1].split('"')[1]
        end = lines[-2].split()[-1].split('"')[1]

        start_time = datetime.strptime(start, '%H:%M:%S')
        end_time = datetime.strptime(end, '%H:%M:%S')
        increment_time =  ((end_time - start_time) / len(bom)).total_seconds()
        bom_basinc_obj_list = []
        kepce_basinc_obj_list = []      
        if i>=1:
            st = segment_kepce[i-1][-1][0]
        else:
            st=0
        for a in range(len(bom)):
            time = st + timedelta(seconds=a * increment_time).total_seconds()
            bom_basinc_obj_list.append(bom_basinc_obj(time, bom[a]))
            kepce_basinc_obj_list.append(kepce_basinc_obj(time, kepce[a]))
        segment_kepce[i] = kepce_basinc_obj_list
        segment_bom[i] = bom_basinc_obj_list
        i+=1  
    return segment_kepce, segment_bom

def read_all_data_and_write():
    i = 0
    while i < 47:
        path = os.path.join('test_dataset', 'kvaser-10.23', f'{i+1}-saha2023-10-23.asc') #test_dataset\kvaser-10.23\1-saha2023-10-23.asc
        with open(path, 'r') as fh:
            data = fh.read().splitlines()
        
        all_data = defaultdict(list)
        breaking_lines = []
        for sample in data[6:-1]:
            line = sample.split()
            if line[2] in significant_types:
                all_data[line[2]].append(line)
            
        for key, value in all_data.items():
            file_name = f'{key}_{i}.txt'
            path = os.path.join('test_dataset' , 'test_can_data' , file_name)
            with open(path, 'w') as fh:
                data = '\n'.join([' '.join(s) for s in value])
                fh.write(data) 
        i+=1

def read_kepce_aci():
    segment_kepce_aci = [[] for i in range(47)]
    i = 0
    while i < 47:
        path = os.path.join('test_dataset', 'test_can_data', f'1E5_{i}.txt')
    #decoder = partial(two_byte, y=10)
        def decoder(x):
            result = two_byte(x, 10)
            aci_value = 50 - result
            return aci_value
        kepce_aci, broken_lines = read_canbus(path, 7, 5, decoder=decoder, data_object=kepce_aci_obj)
        segment_kepce_aci[i] = kepce_aci
        i+=1
    return segment_kepce_aci

if __name__ == "__main__":
    read_bom_kepce()
    #read_all_data_and_write()
    read_kepce_aci()