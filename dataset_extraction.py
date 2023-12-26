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
"""
['0.134390250', '1', 'CF00400x', 'Rx', 'd', '8', 'F0', '91', '91', '8B', '1A', '00', '0F', '91']
"""

bom_basinc_obj = namedtuple('BOM_BASINC', ('time', 'value'))
kepce_basinc_obj = namedtuple('KEPCE_BASINC', ('time', 'value'))

rpm_obj = namedtuple('RPM', ('time', 'value'))
vites_obj = namedtuple('VITES', ('time', 'value'))
bom_aci_obj = namedtuple('BOM_ACI', ('time', 'value'))
kepce_aci_obj = namedtuple('KEPCE_ACI', ('time', 'value'))
valve_obj = namedtuple('VALVE', ('time', 'x_axis', 'y_axis', 'kepce_acma', 'kepce_kapama', 'bom_kald', 'bom_ind'))
lamba = namedtuple('LAMBA', ('time', 'value'))
pedal_volt_obj = namedtuple('VOLT', ('time', 'value'))
km_sa_obj = namedtuple('KM_SA', ('time', 'value'))
fren_obj = namedtuple('FREN', ('time', 'value'))
fuel_level_obj = namedtuple('FUEL', ('time', 'value'))

two_byte = lambda x, y: int(''.join(x), 16) / y
one_byte = lambda x: int(''.join(x), 16)

all_types = {'129', 'CF00300x', '1B190D0Dx', '382', '160', '18F00B38x', '207', '18FEEE00x', 
             '152', '171', '18FEE400x', '158', '18FEF803x', '18FDCD4Ex', 'ErrorFrame', '1EF', 
             '381', '18FEDF00x', '101', '1B24FF0Fx', '1ED', '18F00100x', '571', '1B181212x', 
             '18FEF000x', '1B188D8Dx', '182', '113', '18FEF700x', '1B180F0Fx', '151', '1E5', 
             '155', '18F00B08x', '187', '21C', '282', '156', '118', '1B188585x', '186', 
             '173', '10E', '18EBFF00x', 'CFDCC4Ex', '1B24FF05x', '14FF0117x', '18FEF600x', 
             '1B24FF85x', '177', '18FF1F03x', '10D', '715', '1B180808x', '153', '107', '318', 
             '81', '12C', '18FEFF00x', '108', '18FF2003x', '154', '18FEF200x', '1CFEC349x',
             '157', '1B24FF0Dx', '181', '18FFE517x', '18FEBD00x', '159', '7003305x', '214', 
             '12A', '172', '12B', '106', '18FECA00x', '1B190505x', '1EB', '18FFFF4Ex', 
             '1B24FF08x', '281', '18FEF100x', '18FEE500x', 'CF00203x', '1B180505x', 
             '174', '1E9', 'CF00400x', 'C000003x', '18FDCA00x', '18FEFB10x', '18EA0000x', 
             '21B', '1B180D0Dx', '18FF1E03x', '1B24FF12x', '218', '21A', '18FEEF00x', '286', 
             '1B24FF8Dx', '175', '1E7', '18FEF500x', '18ECFF00x', '18FECA38x', '18F00503x', '418', '176'
             }

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


def read_all_data_and_write():
    path = os.path.join('dataset', 'canbus', '26.102023-10-27_14-12-58.asc')
    with open(path, 'r') as fh:
        data = fh.read().splitlines()
    
    all_data = defaultdict(list)
    breaking_lines = []
    for sample in data[6:-1]:
        line = sample.split()
        if line[2] in significant_types:
            all_data[line[2]].append(line)
        
    for key, value in all_data.items():
        file_name = f'{key}.txt'
        path = os.path.join('dataset' , 'canbus' , file_name)
        with open(path, 'w') as fh:
            data = '\n'.join([' '.join(s) for s in value])
            fh.write(data)   


def sample_bom_kepce(list_of_float, start_time, end_time, start_indeice, end_indice):
    pass


def read_canbus(path, start_indice, end_indice, decoder=None, data_object=None):
    with open(path, 'r') as fh:
        lines = fh.read().split('\n')
    data = []
    broken_lines = []
    for line in lines:
        try:
            fields = line.split()
            value_str = fields[start_indice:end_indice:-1]
            value = decoder(value_str) # get the respective fields for hexadecimal rep and create decimal value
            time = int(1000 * float(fields[0]))
            data.append(data_object(time=time, value=value))
        except:
            traceback.print_exc()
            broken_lines.append(line)
            continue
    return data, broken_lines

def read_rpm():
    path = os.path.join('dataset', 'canbus', 'CF00400x.txt')
    decoder = partial(two_byte, y=8)
    rpm, broken_lines = read_canbus(path, 10, 8, decoder=decoder, data_object=rpm_obj)
    return rpm, broken_lines

def read_fuel():
    path = os.path.join('dataset', 'canbus', '151.txt')
    decoder = partial(two_byte, y=1)
    fuel_level, broken_lines = read_canbus(path, 9, 8, decoder=decoder, data_object=fuel_level_obj)
    return fuel_level, broken_lines

def read_km_sa():
    path = os.path.join('dataset', 'canbus', '18FF1E03x.txt')
    #two_byte = lambda x, y: int(''.join(x), 16) / y
    #decoder = partial(two_byte, y=384.06145)
    def decoder(x):
        result = two_byte(x, 8)
        km_sa_value = (result*2*3.14*0.650*60)/(1000*11.76)
        return km_sa_value
    km_sa, broken_lines = read_canbus(path, 9, 7,decoder=decoder, data_object=km_sa_obj)
    #(2*PI*r_dyn*output_speed_rpm x 60)/(1000 x i_axle) -- Hız (km/sa ) -> ID : 18FF1E03 Byte : 3 - 4
    return km_sa, broken_lines

def read_pedal_volt():
    path = os.path.join('dataset', 'canbus', 'CF00300x.txt')
    #Volt = (int(Byte2)*3,2 )/255 + 1,1 -- Pedal Voltaj -> ID : CF00300  Byte : 2
    def decoder(x):
        result = one_byte(x)
        volt = (result*3.2)/255 + 1.1
        return volt
    pedal_volt, broken_lines = read_canbus(path, 8, 7, decoder=decoder, data_object=pedal_volt_obj)
    return pedal_volt, broken_lines

def read_vites():
    path = os.path.join('dataset', 'canbus', '152.txt')
    vites, broken_lines = read_canbus(path, 7, 6, decoder=one_byte, data_object=vites_obj)
    return vites, broken_lines

def read_fren():
    path = os.path.join('dataset', 'canbus', '152.txt')
    #Fren -> ID : 152 Byte : 7 __ Bit : 4 On/Off 
    def decoder(x):
        result = one_byte(x)
        fren = result & 16
        # return 1 if fren else 0
        return fren>>4
    fren, broken_lines = read_canbus(path, 13, 12, decoder=decoder, data_object=fren_obj)
    return fren, broken_lines

def read_bom_aci():
    path = os.path.join('dataset', 'canbus', '1E7.txt')
    #decoder = partial(two_byte, y=10)
    def decoder(x):
        result = two_byte(x, 10)
        aci_value = 80 - result
        return aci_value
    bom_aci, broken_lines = read_canbus(path, 7, 5, decoder=decoder, data_object=bom_aci_obj)
    return bom_aci, broken_lines

def read_kepce_aci():
    path = os.path.join('dataset', 'canbus', '1E5.txt')
    #decoder = partial(two_byte, y=10)
    def decoder(x):
        result = two_byte(x, 10)
        aci_value = 50 - result
        return aci_value
    kepce_aci, broken_lines = read_canbus(path, 7, 5, decoder=decoder, data_object=kepce_aci_obj)
    return kepce_aci, broken_lines

def read_lamp():
    path = os.path.join('dataset', 'canbus', '153.txt')
    decoder = one_byte
    kepce_aci, broken_lines = read_canbus(path, 6, 5, decoder=decoder, data_object=lamba)
    return kepce_aci, broken_lines

def read_joystick():
    path = os.path.join('dataset', 'canbus', '10D.txt')
    with open(path, 'r') as fh:
        lines = fh.read().split('\n')
    data = []
    broken_lines = []
    for line in lines:
        try:
            fields = line.split()
            x_axis = int(fields[6], 16)
            if x_axis <= 127:
                kepce_acma = x_axis*(100/127)
            elif x_axis > 127:
                kepce_kapama = 100 - (100/127)*(x_axis & 0x7f)
            y_axis = int(fields[7], 16)
            if y_axis <= 127:
                bom_ind = 100 - y_axis*(100/127)
            elif y_axis > 127:
                bom_kald = 100 - (100/127) * (y_axis & 0x7f)
            time = int(1000 * float(fields[0]))
            data.append(valve_obj(time=time, x_axis=x_axis, y_axis=y_axis, kepce_acma=kepce_acma, kepce_kapama=kepce_kapama, bom_ind=bom_ind, bom_kald=bom_kald))
        except:
            broken_lines.append(line)
            continue
    return data, broken_lines

def register_row(df, data_obj, field, column):
    if data_obj is not None:
        row = data_obj.time // 20
        if row > 0:
            try:
                value = getattr(data_obj, field)
                df.iloc[row][column] = value
            except:
                print("hey")

def read_bom_kepce(canbus_date):
    path = os.path.join('dataset', 'datalogger', '1412_261023_1412.txt')
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
    for i in range(len(bom)):
        time = int(1000 * (start_time - canbus_date + timedelta(seconds=i * increment_time)).total_seconds())
        bom_basinc_obj_list.append(bom_basinc_obj(time, bom[i]))
        kepce_basinc_obj_list.append(kepce_basinc_obj(time, kepce[i]))
    return kepce_basinc_obj_list, bom_basinc_obj_list

def get_datetime_for_canbus():
    path = os.path.join('dataset', 'canbus', '26.102023-10-27_14-12-58.asc')
    date = path.split('_')[1].split('.')[0]
    return datetime.strptime(date.replace('-', ':'), '%H:%M:%S')

def create_data_frame():
    # TODO: add time stamp column, will be needed when we add basinc verileri
    canbus_date = get_datetime_for_canbus()
    full_kepce_basinc, full_bom_basinc = read_bom_kepce(canbus_date)
    full_rpm, _ = read_rpm()
    full_km_sa, _ = read_km_sa()
    full_pedal_volt, _ = read_pedal_volt()
    full_fren, _ = read_fren()
    full_vites, _ = read_vites()
    full_bom_aci, _ = read_bom_aci()
    full_kepce_aci, _ = read_kepce_aci()
    full_valve, _ = read_joystick()
    full_lamba, _ = read_lamp()
    full_fuel_level, _ = read_fuel()
    ls = list(map(lambda x: x[-1].time // 20, [full_kepce_aci, full_bom_aci, full_valve, full_vites, full_rpm, full_km_sa, full_pedal_volt, full_fren, full_fuel_level]))
    print(ls)
    length = max(ls)
    df = pd.DataFrame(np.nan, index=range(length+1), columns=['lamba', 'kepce_basinc','bom_basinc', 'kepce_aci', 'bom_aci', 'vites', 'fren', 'valve_x', 'valve_y','kepce_acma', 'kepce_kapama','bom_ind', 'bom_kald', 'rpm','km_sa', 'pedal_volt', 'fuel_level'])
    for lamba, kepce_basinc, bom_basinc, kepce_aci, bom_aci, vites,fren, valve, rpm , km_sa, pedal_volt, fuel_level in zip_longest(full_lamba, full_kepce_basinc, full_bom_basinc, full_kepce_aci, full_bom_aci, full_vites, full_fren, full_valve, full_rpm, full_km_sa, full_pedal_volt, full_fuel_level):
        # add canbus data into dataframe
        register_row(df, lamba, 'value', 'lamba')
        register_row(df, kepce_aci, 'value', 'kepce_aci')
        register_row(df, bom_aci, 'value', 'bom_aci')
        register_row(df, vites, 'value', 'vites')
        register_row(df, fren, 'value', 'fren')
        register_row(df, rpm, 'value', 'rpm')
        register_row(df, km_sa, 'value', 'km_sa')
        register_row(df, pedal_volt, 'value', 'pedal_volt')
        register_row(df, valve, 'x_axis', 'valve_x')
        register_row(df, valve, 'y_axis', 'valve_y')
        register_row(df, valve, 'kepce_acma', 'kepce_acma')
        register_row(df, valve, 'kepce_kapama', 'kepce_kapama')
        register_row(df, valve, 'bom_ind', 'bom_ind')
        register_row(df, valve, 'bom_kald', 'bom_kald')
        register_row(df, fuel_level, 'value', 'fuel_level' )
        # add datalogger data into dataframe
        register_row(df, kepce_basinc, 'value', 'kepce_basinc')
        register_row(df, bom_basinc, 'value', 'bom_basinc')
        
    # Fill rows with NaN
    df = df.fillna(method='ffill',axis=0)
    # sample to 100ms from 20ms
    df = df.groupby(np.arange(len(df))//5)[['lamba', 'kepce_basinc','bom_basinc', 'kepce_aci', 'bom_aci', 'vites','fren', 'valve_x', 'valve_y','kepce_acma', 'kepce_kapama', 'bom_ind', 'bom_kald', 'rpm', 'km_sa', 'pedal_volt', 'fuel_level']].median()
    
    # TODO remove rows with NaN

    return df

def create_segments(df):
    segments = [[]]
    curr_state = df.iloc[0]['lamba']
    for index, row in df.iterrows():
        if row['lamba'] != curr_state:
            segments.append([])
            curr_state = row['lamba']
        segments[-1].append(row)
    segments = [pd.DataFrame(segment) for segment in segments]
    return segments

def AmazingModel(*arg):
    pass

def get_x_y(segment: pd.DataFrame):
    # create input to the model
    x = [[s.kepce_basinc, s.bom_basinc, s.kepce_aci, s.bom_aci] for s in segment.iterrows()]
    X = np.array(x).T
    mode=1
    # create output of the inputs created above
    if mode==1:
        y = [[s.valve_x, s.valve_y, s.pedal_volt] for _,s in segment.iterrows()]
        Y = np.array(y[1:]).T
    else:
        y = [[s.kepce_kapama, s.bom_kald, s.pedal_volt] for _,s in segment.iterrows()]
        Y = np.array(y[1:]).T
        
    return X, Y

def get_training_data():
    df = create_data_frame()
    segments = create_segments(df)
    print(f'total numbr of segments is: {len(segments)}')
    filling_segments = [s for s in segments[2:-1] if s.iloc[0].lamba == 2]
    print(f'total numbr of filling segments is: {len(segments)}')

    return filling_segments

if __name__ == "__main__":
    step = 1
    # step-1
    if step == 0:
        read_all_data_and_write()
    df = get_training_data()