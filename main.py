import SynPy as sp
import os
import numpy as np

#--Run Options--#
conf_dir = os.path.join(os.getcwd(), 'confs/dosage/')
output_dir = os.path.join(os.getcwd(), 'outputs/dosage/')



for dose in range(1000, 1040 + 20, 20):
    grid_dir_name = f'bursts_oscillation_{dose}'
    new_conf_dir = os.path.join(conf_dir, grid_dir_name)
    new_output_dir = os.path.join(output_dir, grid_dir_name)

    # Run 
    params = { # Replaces each dictionary key with the corresponding value in the .conf
        'Onset:': 150,
#         'Interval' : .5e-2,
    #     'Time:' : 600,
    #     'Duration' : 200,
    #     'Bursts' : 3,
    #     'Oscillation Frequency' : 6

    #     'Propagator 3|Tau' : .055,
    #     'Propagator 7|Tau' : .055,
    #     'Propagator 8|Tau' : .055,
    #     'Propagator 10|Tau' : .055,
    }

    perm_dict = {
    #     'Bursts' : [2,3,1],
        'Bursts' : [2,20,1],
    #     'Burst Frequency' : [40, 60, .25],
    #     'Oscillation Frequency' : [4,8,.1],
        'Oscillation Frequency' : [1,20,.25],
    #     'Duration' : [100,1000,10]
    }

    sp.dot_conf('eirs-tms-custom.conf').grid_outputs(perm_dict, new_conf_dir, new_output_dir, params, dynamic_dose = dose)
