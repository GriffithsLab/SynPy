import os
os.environ['OMP_NUM_THREADS'] = '1' # os and this line need to be imported before *any* other packages, including SynPy
import SynPy as sp
import numpy as np
import time
import pickle
import shutil

#--Run Options--#
conf_dir = os.path.join(os.getcwd(), 'confs/dosage/')
output_dir = os.path.join(os.getcwd(), 'outputs/dosage/')

pkl = "scaled_amp_105_limit_with_gnmda.pkl"

pulse_dose_range = {'start' : 600,
                    'stop' : 600,
                    'step' : 20}




purge_dir = True
purge_dict = True


def check_and_create_pkl(pkl_filename):
    """
    Check if a pickle file exists in the current working directory.
    If it does not exist, create a new pickle file with an empty dictionary.
    """
    if not os.path.isfile(pkl_filename):
        with open(pkl_filename, 'wb') as file:
            pickle.dump({}, file)





check_and_create_pkl(pkl)
if purge_dict:
    with open(pkl, 'wb') as file:
        pickle.dump({}, file)
with open(pkl, 'rb') as file:
    pickle.load(file)

        


for dose in range(pulse_dose_range['start'], 
                  pulse_dose_range['stop'] + pulse_dose_range['step'], 
                  pulse_dose_range['step']):
    
    grid_dir_name = f'bursts_oscillation_{dose}'
    new_conf_dir = os.path.join(conf_dir, grid_dir_name)
    new_output_dir = os.path.join(output_dir, grid_dir_name)

    # Run 
    params = { # Replaces each dictionary key with the corresponding value in the .conf
        'Onset:': 150,
    }

    perm_dict = {
        'Bursts' : [2,20,1],
        'Oscillation Frequency' : [1,20,.25],
    }
    
    expected_file_count = len(sp.valid_iTBS_protocols())
    
    def submit_jobs():
        
        if purge_dir:
            if os.path.exists(new_conf_dir):
                try:
                    shutil.rmtree(new_conf_dir)
                except: pass
            if os.path.exists(new_output_dir):
                try:
                    shutil.rmtree(new_output_dir)
                except: pass
                

        # Make the confs and submit them for batch jobs
        num_submitted_jobs = sp.dot_conf('eirs-tms-custom.conf').grid_outputs(perm_dict, 
                                                         new_conf_dir, 
                                                         new_output_dir, 
                                                         params, 
                                                         dynamic_dose = dose,
                                                         dynamic_amp = 105,
                                                         filtered_perms = True)



        while True:
            complete_jobs = sp.list_files(new_output_dir, extension_filter = "done.txt")
            complete_output_files = sp.list_files(new_output_dir, extension_filter = ".output")

            if len(complete_jobs) >= num_submitted_jobs:
                break
            else:
                print(f'{len(complete_output_files)}/{expected_file_count} .output files written', end='; ') 
                print(f'{len(complete_jobs)}/{num_submitted_jobs} jobs fully complete')
                time.sleep(10)
    
    def load_outputs():

        with open(pkl, 'rb') as file:
            xyz_gain_dict = pickle.load(file)

        df = sp.perm_load(new_output_dir).perm_df(load_type = 'parallel')

        print(df)

        xyz_gain_dict[grid_dir_name] = df

        # Open the same file in binary write mode to save the modified data
        with open(pkl, 'wb') as file:
            pickle.dump(xyz_gain_dict, file)

        if len(df.index) != expected_file_count:
            missing_entries = [entry for entry in sp.valid_iTBS_protocols() if entry not in df.index]
            raise Exception(f'Write error; missing protocol entries: \n{missing_entries}')
            
    submit_jobs()
    load_outputs()
        
#     shutil.rmtree(new_conf_dir)
#     shutil.rmtree(new_output_dir)
