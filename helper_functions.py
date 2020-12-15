# -*- coding: utf-8 -*-
"""
Helper Functions for computing connectivity matrix

@author: Ben + Cody
"""


#create neuron class
import pandas as pd
import numpy as np
class neuron:
    '''
    object: neuron
    name=EIP0
    neuron_class = EIP
    postsynaptic_buton=list(PB-R8,'PB-R9'...etc) /1
    presynaptic_spine=list(PB-R8,'PB-R9'...etc)  /2
    '''
    def __init__(self,name = '',neuron_class = '',postsynaptic = list(),presynaptic = list()):
        self.name=name
        self.neuron_class=neuron_class
        self.postsynaptic=postsynaptic
        self.presynaptic=presynaptic
        
#import connectivity matrix from paper and fill NaNs with 0s
def read_connectivity_matrix(connectivity_matrix):
    #current filename is connectivity_matrix.csv
    
    df = pd.read_csv(connectivity_matrix)
    df = df.fillna(value=0)
    n_types = df['Neuron Type']
    n_class = df['Neuron Class']
    postsynaptic_connections = list()
    presynaptic_connections = list()
    neurons = []
    for num in range(0,len(n_types)):
        row_values = df.values[num]
        
        #find 1s
        one_locs = np.where(df.values[num]==1)
        #find 2s
        two_locs = np.where(df.values[num]==2)  
        
        n = neuron(name=df['Neuron Type'][num], neuron_class=df['Neuron Class'][num],
                postsynaptic=df.columns[one_locs],presynaptic=df.columns[two_locs])
        neurons.append(n)
    return neurons
#read in the inhibitory/excitatory neurons from the paper, store as df_out:
def read_inhib_matrix(inhibitory_excitatory):
    df=pd.read_csv(inhibitory_excitatory)
    dfvals = df.values
    mat = dfvals[2:,2:]
    source_list=df.values[1,2:]
    destination_list=df.values[2:,1]
    [one_rows,one_cols]=np.where(mat=='1')
    [neg_rows,neg_cols]=np.where(mat=='-1')
    sources = source_list[np.concatenate((one_rows,neg_rows))].tolist()
    destinations = destination_list[np.concatenate((one_cols,neg_cols))].tolist()
    ones_list = np.ones((1,len(one_rows)))[0]
    neg_list = np.ones((1,len(neg_rows)))*(-1)
    ones_list = ones_list.tolist()
    neg_list = neg_list.tolist()
    ones_list.extend(neg_list[0])
    fixed_sources = list()
    fixed_destinations = list()
    for s in sources:
        fixed_sources.append(s.replace(' ',''))
    for d in destinations:
        fixed_destinations.append(d.replace(' ',''))
    df_dict = {'source':fixed_sources,'destination':fixed_destinations,'con_type':ones_list}
    df_out = pd.DataFrame(df_dict)
    return df_out  
