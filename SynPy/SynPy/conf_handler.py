from .nftsim_generator_FUNCTIONS import *

class dot_conf:
    def __init__(self, conf_path):
        self.conf_path = conf_path
        self.f_name = os.path.basename(self.conf_path)

    def gen_confs(self, perm_dict, new_conf_dir, params = {}, write_confs = True):
        """
        Given a template .conf file and a dictionary object containing parameter values ranges to be iterated upon, 
        generates a new .conf for each unique permutation of parameters.  Writes each .conf file to disk by default.
        """

        with open(self.conf_path, 'r') as template_conf:
            conf_txt = template_conf.readlines()

        [update_param(kw, value, conf_txt) for kw, value in params.items() if len(params) > 0] # Alter base parameters before iterating 
        print()

        new_confs = {} # will take the format {string_name : content}

        perms = generate_permutations(perm_dict) # list of all unique parameter permutations from perm_dict

        for perm in perms:

            # update the conf_txt with each param in the unique permuatation item        
            [update_param(list(perm_dict.keys())[p_idx], param, conf_txt) for p_idx, param in enumerate(perm)]

             # list of key: param_value pair, per perm
            idv_params = [f'{key[:3]}={perm[k_idx]}' for k_idx, key in enumerate(list(perm_dict.keys()))]

            param_string = '_'.join(idv_params) # join them into one string, separated by '_' 
            perm_string = f"{str(self.f_name.split('.')[0])}_{'PERM'}_[{param_string.lower()}].conf" # full string for a single permuatation

            # make dictionary perm_string [key] correspond to the conf_txt contents with all parameter changes of the permutation
            new_confs[perm_string] = conf_txt[:]

        # Names the dir for the new confs based on perm_dict keys
        new_confs_dir_string = f'{"_".join([k.lower() for k in perm_dict.keys()])}/'.replace(':', '')
        
        if write_confs:
            save_confs(new_confs, conf_dir)

        return len(new_confs)
    
    
    def grid_outputs(self, perm_dict, new_conf_dir, new_output_dir, params = {}, batch = True, nft_path = 'nftsim/bin/nftsim'):
        """
        Given a conf file path, a permutation dictionary, and new conf/output directory names, creates and writes each permutation 
        of the .conf file, then generates each corresponding .output file.
        
        perm_dict: 
        ex. perm_dict = {'Bursts' : [2,20,1],
                         'Oscillation' : [1,20,.25]}
                         
        batch (optional; default: True):
            If True, submits jobs to process .conf files into .outputs through sbatch SLURM scheduler.
            If False, individually processes each .conf file through NFTsim.
        """

        grid_points = self.gen_confs(self.conf_path, new_conf_dir, perm_dict, params)

        gen_outputs(new_conf_dir, new_output_dir, batch, nft_path)

        # Detect when the number of files in the output dir are equal to the total files generated
        while True:
            num_outputs = len(list_files(new_output_dir, extension_filter = '.output'))
            if num_outputs >= grid_points:
                print(f'All {num_outputs} files written to {new_output_dir}')
                break
            else:
                print(f'{int(num_outputs)}/{int(grid_points)} files')
                time.sleep(10)