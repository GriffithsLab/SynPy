import SynPy as sp
import os
import numpy as np
import time
import pickle
import shutil

#--Run Options--#
conf_dir = os.path.join(os.getcwd(), 'confs/dosage/')
output_dir = os.path.join(os.getcwd(), 'outputs/dosage/')

pkl = "xyz_gain_dict_600-3000.pkl"

pulse_dose_range = {'start' : 1100,
                    'stop' : 1140,
                    'step' : 20}

for dose in range(pulse_dose_range['start'], 
                  pulse_dose_range['stop'] + pulse_dose_range['step'], 
                  pulse_dose_range['step']):
    
    grid_dir_name = f'bursts_oscillation_{dose}'
    new_conf_dir = os.path.join(conf_dir, grid_dir_name)
    new_output_dir = os.path.join(output_dir, grid_dir_name)

    # Run 
    params = { # Replaces each dictionary key with the corresponding value in the .conf
        'Onset:': 150,
        'Interval' : 1e-2
    }

    perm_dict = {
        'Bursts' : [2,20,1],
        'Oscillation Frequency' : [1,20,.25],
    }

    # Make the confs and submit them for batch jobs
    sp.dot_conf('eirs-tms-custom.conf').grid_outputs(perm_dict, 
                                                     new_conf_dir, 
                                                     new_output_dir, 
                                                     params, 
                                                     dynamic_dose = dose, 
                                                     filtered_perms = True)
    
    # Don't run the next segment of code until all batch jobs are finished processing
    expected_file_count = len(sp.valid_iTBS_protocols())
    while True:
        num_outputs = len(sp.list_files(new_output_dir, extension_filter = '.output'))
        if num_outputs >= expected_file_count:
            break
        else:
            print(f'{int(num_outputs)}/{int(expected_file_count)} .output files')
            time.sleep(10)

    with open(pkl, 'rb') as file:
        xyz_gain_dict = pickle.load(file)

    xyz_gain_dict[grid_dir_name] = sp.perm_load(new_output_dir).perm_df(load_type = 'parallel')

    # Open the same file in binary write mode to save the modified data
    with open(pkl, 'wb') as file:
        pickle.dump(xyz_gain_dict, file)


    shutil.rmtree(new_conf_dir)
    shutil.rmtree(new_output_dir)
    
    
    
