import os
from .nftsim_generator_FUNCTIONS import (
    param_value, 
    update_param, 
    generate_permutations, 
    tbs_pulse_time, 
    save_confs, 
    nftsim_run
)
from .output_handler import dot_output
from .batch_jobs import gen_outputs

class dot_conf:
    def __init__(self, conf_path):
        self.conf_path = os.path.abspath(conf_path)
        self.f_name = os.path.basename(self.conf_path)
        if self.f_name.split('.')[-1] == 'output':
            raise Exception('dot_conf does not manage .output files.  Pass a .conf file instead.') 
        
        with open(self.conf_path, 'r') as f:
            self.conf_text = f.readlines()

    def gen_confs(self, perm_dict, new_conf_dir, params = {}, dynamic_dose = None, write_confs = True):
        """
        perm_dict -- Dictionary object containing n parameters and their corresponding value ranges to generate permuations for.
                     Formatted as {'PARAMETER NAME' : [LOWEST VALUE, HIGHEST VALUE, STEP SIZE]}
                     
        new_conf_dir -- Directory to write the .conf permutations to.  Will create the directory if it does not exist.
        
        params (optional) -- Dictionary object to modify base parameters.
                             Formatted as {'PARAMETER NAME' : NEW VALUE}
                             
        write_confs -- True/False write permutations to disk.  Useful for first checking if permutations look correct before creating files
        
        Given a template .conf file and a dictionary object containing parameter values ranges to be iterated upon, 
        generates a new .conf for each unique permutation of parameters.  Writes each .conf file to disk by default.
        """

        new_conf_text = self.conf_text
        if params:
            [update_param(kw, value, new_conf_text, verbose = False) for kw, value in params.items()] # Alter base params before iterating

        new_confs = {} # will take the format {f_name : conf_txt}

        perms = generate_permutations(perm_dict) # list of all unique parameter permutations from perm_dict
        keywords = list(perm_dict.keys())

        for perm in perms:

            # update the new_conf_text with each param in the unique permuatation item
            [update_param(keywords[p_idx], format(param, '.2f'), new_conf_text) for p_idx, param in enumerate(perm)]
            ### Adjusts stim length based on desired number of pulses ###
            if dynamic_dose:
                stim_len = tbs_pulse_time(dynamic_dose, 
                                          pulses_per_burst = float(param_value('Bursts', new_conf_text)), 
                                          inter_burst_freq = float(param_value('Oscillation Frequency', new_conf_text)))
                update_param('Duration', format(stim_len, '.2f'), new_conf_text)
                
                onset = float(param_value('Onset', new_conf_text))
                sim_len = onset + stim_len + 100 + onset
                update_param('Time', format(sim_len, '.2f'), new_conf_text)
            #############################################################
                
            var_time = float(param_value('Time', new_conf_text))
            var_onset = float(param_value('Onset', new_conf_text))
            var_duration = float(param_value('Duration', new_conf_text))

            if var_time < (var_onset + var_duration): # Check that the stim onset + duration does not exceed the total simulation length
                raise ValueError('Parameter error: Stimulation onset + duration length > simulation length (time).') # Halt the process

             # list of key: param_value pair, per perm
            idv_params = [f'{key[:3]}={perm[k_idx]:.2f}' for k_idx, key in enumerate(keywords)]

            param_string = '_'.join(idv_params) # join them into one string, separated by '_' 
            perm_string = f"{str(self.f_name.split('.')[0])}_{'PERM'}_[{param_string.lower()}].conf" # full string for a given permuatation

            # make dictionary perm_string [key] correspond to the new_conf_text contents with all parameter changes of the permutation
            new_confs[perm_string] = new_conf_text[:]

        # Names the dir for the new confs based on perm_dict keys
        new_confs_dir_string = f'{"_".join([k.lower() for k in perm_dict.keys()])}/'.replace(':', '')
        
        if write_confs:
            save_confs(new_confs, new_conf_dir)

        return len(new_confs)
    
    
    def grid_outputs(self, perm_dict, new_conf_dir, new_output_dir, params = {}, 
                     
                     dynamic_dose = None, 
                     batch = True, 
                     nft_path = 'nftsim/bin/nftsim'):
        """
        Given a conf file path, a permutation dictionary, and new conf/output directory names, creates and writes each permutation 
        of the .conf file, then generates each corresponding .output file.
        
        perm_dict -- Dictionary object containing n parameters and their corresponding value ranges to generate permuations for.
                     Formatted as {'PARAMETER NAME' : [LOWEST VALUE, HIGHEST VALUE, STEP SIZE]}
                     
        new_conf_dir -- Directory to write the .conf permutations to.  Will create the directory if it does not exist.
        
        new_output_dir -- Directory to write the corresponding .output permutations to.  Will create the directory if it does not exist.
        
        params (optional) -- Dictionary object to modify base parameters.
                             Formatted as {'PARAMETER NAME' : NEW VALUE}
                             
        batch (optional; default: True):
            If True, submits jobs to process .conf files into .outputs through sbatch SLURM scheduler.
            If False, individually processes each .conf file through NFTsim.
        """

        grid_points = self.gen_confs(perm_dict, new_conf_dir, params, dynamic_dose)

        gen_outputs(new_conf_dir, new_output_dir, batch, nft_path)

        print('Jobs submitted.')
        # Detect when the number of files in the output dir are equal to the total files generated
#         while True:
#             num_outputs = len(list_files(new_output_dir, extension_filter = '.output'))
#             if num_outputs >= grid_points:
#                 print(f'All {num_outputs} files written to {new_output_dir}')
#                 break
#             else:
#                 print(f'{int(num_outputs)}/{int(grid_points)} files')
#                 time.sleep(10)
                
    def run(self, conf_dir = 'confs/', output_dir = 'outputs/', params = {},
            gains = False, # return CT gain values instead of individual fields
            normalize = False, # return df with normalized column values
            _name_field_ = True, # alter column (field) names to replace numbers with letters
            nftsim_path = 'nftsim/bin/nftsim'):
        
        
        output_name = self.f_name.replace('.conf', '.output')
        output_path = os.path.join(output_dir, output_name)

        # Update the value of each parameter within the passed 'params' dictionary, if the dict contains items
        new_conf_text = self.conf_text
        
        if params:
            [update_param(kw, value, new_conf_text) for kw, value in params.items()]
        
        # Save a new .conf file, which may or may not have been modified by params, from which to generate the .output file
        save_confs(new_confs = {self.f_name: new_conf_text[:]},
                   conf_dir = conf_dir)

        # Generate .output file from .conf file
        nftsim_run(nftsim_path = nftsim_path, 
                   conf_path = os.path.join(conf_dir, self.f_name), 
                   output_path = output_path)

        return dot_output(output_path).df(gains, normalize, _name_field_)