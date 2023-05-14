from .nftsim_generator_FUNCTIONS import *
from .output_handler import dot_output

class dot_conf:
    def __init__(self, conf_path):
        self.conf_path = conf_path
        self.f_name = os.path.basename(self.conf_path)
        
        with open(self.conf_path, 'r') as f:
            self.conf_text = f.readlines()

    def gen_confs(self, perm_dict, new_conf_dir, params = {}, write_confs = True):
        """
        Given a template .conf file and a dictionary object containing parameter values ranges to be iterated upon, 
        generates a new .conf for each unique permutation of parameters.  Writes each .conf file to disk by default.
        """

        new_conf_text = self.conf_text
        if params:
            [update_param(kw, value, new_conf_text) for kw, value in params.items()] # Alter base params before iterating 

        new_confs = {} # will take the format {f_name : conf_txt}

        perms = generate_permutations(perm_dict) # list of all unique parameter permutations from perm_dict
        keywords = list(perm_dict.keys())

        for perm in perms:

            # update the new_conf_text with each param in the unique permuatation item
            [update_param(keywords[p_idx], param, new_conf_text) for p_idx, param in enumerate(perm)]

             # list of key: param_value pair, per perm
            idv_params = [f'{key[:3]}={perm[k_idx]}' for k_idx, key in enumerate(keywords)]

            param_string = '_'.join(idv_params) # join them into one string, separated by '_' 
            perm_string = f"{str(self.f_name.split('.')[0])}_{'PERM'}_[{param_string.lower()}].conf" # full string for a given permuatation

            # make dictionary perm_string [key] correspond to the new_conf_text contents with all parameter changes of the permutation
            new_confs[perm_string] = new_conf_text[:]

        # Names the dir for the new confs based on perm_dict keys
        new_confs_dir_string = f'{"_".join([k.lower() for k in perm_dict.keys()])}/'.replace(':', '')
        
        if write_confs:
            save_confs(new_confs, new_conf_dir)

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

        grid_points = self.gen_confs(perm_dict, new_conf_dir, params)

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
                
    def run(self, conf_dir = 'confs/', output_dir = 'outputs/', params = {},
            gains = False,
            normalize = False,
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
                   output_path = output_path,
                   load_gcc = True)

        return dot_output(output_path).df(gains, normalize)