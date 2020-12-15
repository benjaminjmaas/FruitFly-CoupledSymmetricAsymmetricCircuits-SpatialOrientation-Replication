# -*- coding: utf-8 -*-
"""
Ta_Shun_PythonScript
@author: Ben + Cody
"""

from neurokernel.tools.logging import setup_logger
import sys
import numpy as np
import h5py
import networkx as nx
import pandas as pd
import os
os.chdir('/home/e6095/ffbo/neurodriver/examples/generic/data')
sys.path.append("/home/e6095/ffbo/neurodriver/examples/generic/data")
sys.path.append("/home/e6095/ffbo/neurodriver/examples/generic")
# sys.path.append("/home/e6095/ffbo/neurokernel/")

# Sends outputted graph to a GEXF file
# from Ta_Shun_LPU_Making import create_lpu

# Creates an HDF5 file containing input signals for the specified number of neurons
# The signals consist of a rectangular pulse of specified duration and magnitude
from Ta_Shun_LPU_Making import create_input

# Define variables for the LPU
lpu_file_name = 'generic_lpu.gexf.gz'                     #File to write the LPU graph to
lpu_name = 'Ta_Shun'                                      #Name of the LPU
neu_num = [4, 49]  #Number of sensory and projection neurons


# Create the LPU
"""
create_lpu(file_name = lpu_file_name,
           lpu_name = lpu_name,
           N_sensory = neu_num[0],
           N_proj = neu_num[1])
"""


# Define variables for the input

in_file_name = 'Ta_Shun_input.h5'  #File to output the generated input data to
N_sensory = neu_num[0]             #Number of sensory neurons
dt = 1e-3                         #Time resolution of generated signal
dur = 10.0                          #Duration of generated signal
start = 0.3                        #Start time of signal pulse
stop = 0.6                         #Stop time of signal pulse
I_max = 0.6                        #Pulse magnitude

# Create the input data
create_input(file_name = in_file_name,
             N_sensory = N_sensory,
             dt = dt,
             dur = dur,
             start = start,
             stop = stop,
             I_max = I_max)

g = nx.read_gexf('Ta_Shun_LPU.gexf.gz')
bad_nodes=list()
[n for n in g.nodes(data=True) if 'class' not in n[1]]

print('number of nodes: ',len(g.nodes(data=True)))
bad_nodes_num=list()
for num,n in zip(range(0,len(g.nodes(data=True))),g.nodes(data=True)):
    if 'class' not in n[1]:
        bad_nodes.append(num)

        
#visualize the connections
# get_ipython().run_line_magic('matplotlib', 'inline')
nx.draw(g)


[n for n in g.nodes(data=True) if 'class' not in n[1]]

if __name__ == '__main__':
    import argparse
    import itertools
    import networkx as nx
    from neurokernel.tools.logging import setup_logger
    import neurokernel.core_gpu as core

    from neurokernel.LPU.LPU import LPU

    from neurokernel.LPU.InputProcessors.FileInputProcessor import FileInputProcessor
    from neurokernel.LPU.InputProcessors.StepInputProcessor import StepInputProcessor
    from neurokernel.LPU.OutputProcessors.FileOutputProcessor import FileOutputProcessor

    import neurokernel.mpi_relaunch


    #Define variables for the process
    lpu_file_name = 'Ta_Shun_LPU.gexf.gz'  #File where the LPU graph is stored
    in_file_name = 'Ta_Shun_input.h5'      #File where the input data is stored
    dt = 1e-3                              #Time resolution of generated signal
    dur = 10.0                              #Duration of generated signal
    gpu_dev = 0                            #GPU device number
    steps = int(dur/dt)                    #Number of steps
    out_file_name = 'new_Ta_Shun_2.h5'        #File to send the output data to

    #Create a process manager
    man = core.Manager()

    #Parse the LPU from GEXF file
    (comp_dict, conns) = LPU.lpu_parser(lpu_file_name)

    #Input data from HDF5 input file
    my_inputs = []
    #cue on
    my_inputs.append(StepInputProcessor('I', ['PEI2','PEN3','PEI11','PEN10'], 10.0, 0.1, 9.0))
    
    #Cue used for adding second, smaller stimulus
    #my_inputs.append(StepInputProcessor('I', ['PEI5','PEN6','PEI14','PEN15'], 0.10, 0.1, 9.0))
    #cue off
    
    my_inputs.append(StepInputProcessor('I', ['R_PEN'], 10.0, 0.1, 4.0))
    
    #provide input to shift circuit
    my_inputs.append(StepInputProcessor('I', ['R_PEI'], 10.0, 4.0, 8.2))
    my_inputs.append(StepInputProcessor('I', ['PEI2','PEN3'], 10.0, 4.0, 4.4))
    my_inputs.append(StepInputProcessor('I', ['PEI3','PEN4'], 10.0, 4.1, 4.8))
    my_inputs.append(StepInputProcessor('I', ['PEI4','PEN5'], 10.0, 4.5, 5.2))
    my_inputs.append(StepInputProcessor('I', ['PEI5','PEN6'], 10.0, 4.9, 5.6))
    my_inputs.append(StepInputProcessor('I', ['PEI6','PEN7'], 10.0, 5.3, 6.2))
    my_inputs.append(StepInputProcessor('I', ['PEI6','PEN7'], 10.0, 5.9, 6.6))
    my_inputs.append(StepInputProcessor('I', ['PEI5','PEN6'], 10.0, 6.3, 7.0))
    my_inputs.append(StepInputProcessor('I', ['PEI4','PEN5'], 10.0, 6.7, 7.4))
    my_inputs.append(StepInputProcessor('I', ['PEI3','PEN4'], 10.0, 6.9, 7.8))
    my_inputs.append(StepInputProcessor('I', ['PEI2','PEN3'], 10.0, 7.3, 8.2))
    my_inputs.append(StepInputProcessor('I', ['PEI2','PEN3'], 10.0, 7.9, 10.0))
    
    #Output data to HDF5 file
    fl_output_processor = FileOutputProcessor([('V',None),('spike_state',None)], out_file_name, sample_interval=1)

    #Add the LPU to the Manager and attach input and output processors to it
    man.add(LPU, 'ge', dt, comp_dict, conns,
            device = gpu_dev, input_processors = my_inputs,
            output_processors = [fl_output_processor], debug = True)

    logger = setup_logger(file_name="neurokernel.log", screen=True)
    """
    #Use mpi_run_manager to execute the manager in notebook
    from subprocess import CalledProcessError
    try:
        output = mpi_run_manager(man, steps = steps, log = True)
    except CalledProcessError:
        with open('neurokernel.log', 'r') as f:
            print(f.read())
    print(output)
    """

    #This should be replaced by the following in regular code
    man.spawn()
    man.start(steps=steps)
    man.wait()




    from helper_functions import read_connectivity_matrix
    neurons=read_connectivity_matrix('connectivity_matrix.csv')



    j=0
    neurons[0].name
    neurons[0].neuron_class


    import matplotlib as mpl
    mpl.use('agg')
    import matplotlib.pyplot as plt

    import neurokernel.LPU.utils.visualizer as vis
    import networkx as nx
    import h5py

    # Temporary fix for bug in networkx 1.8:
    nx.readwrite.gexf.GEXF.convert_bool = {'false':False, 'False':False,
                                           'true':True, 'True':True}

    # Select IDs of spiking projection neurons:
    G = nx.read_gexf(lpu_file_name)
    neu_proj = sorted([k for k, n in G.node.items() if                    n['name'][:4] == 'proj' and                    n['class'] == 'LeakyIAF'])

    in_uid = 'sensory_0'

    N = len(neu_proj)

    V = vis.visualizer()
    #V.add_LPU(in_file_name, LPU='Sensory', is_input=True)
    #V.add_plot({'type':'waveform', 'uids': [[in_uid]], 'variable':'I'},
    #            'input_Sensory')
    R = ['R_EIP','R_PEI','R_PEN']
    V.add_LPU(out_file_name,  'Generic LPU',
              gexf_file= lpu_file_name)
    V.add_plot({'type':'raster', 'uids': [['EIP'+str(i) for i in range(18)] + ['PEI'+str(i) for i in range(16)] + ['PEN'+str(i) for i in range(16)]+R], 'variable': 'spike_state',
                'yticks': range(1, 1+N),
                'yticklabels': neu_proj, 'title': 'Output'},
                'Generic LPU')

    V.rows = 2
    V.cols = 1
    V.fontsize = 8
    V.xlim = [0, 1.0]

    gen_video = False
    if gen_video:
        V.out_filename = 'generic_output.mp4'
        V.codec = 'mpeg4'
        V.run()
    else:
        V.update_interval = None
        V.run('generic_output_Shift1.png')




    import h5py
    import matplotlib
    matplotlib.use('PS')
    import matplotlib.pyplot as plt

    f = h5py.File('new_Ta_Shun_2.h5')
    t = np.arange(0, steps)*dt

    plt.figure()
    plt.plot(t,list(f['V'].values())[0])
    plt.xlabel('time, [s]')
    plt.ylabel('Voltage, [mV]')
    plt.title('Our model')
    plt.savefig('model_thingy_Shift2.png',dpi=300)
