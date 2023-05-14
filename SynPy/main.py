from SynPy import *

#--Run Options--#
conf_dir = os.path.join(os.getcwd(), 'confs/')
output_dir = os.path.join(os.getcwd(), 'outputs/')


grid_dir_name = 'bursts_oscillation_hr/'
new_conf_dir = os.path.join(conf_dir, grid_dir_name)
new_output_dir = os.path.join(output_dir, grid_dir_name)

# Run 
params = { # Replaces each dictionary key with the corresponding value in the .conf
    'Onset:': 150,
    'Duration:': 200,
    'Time:' : 600
}


perm_dict = {'Bursts' : [2,20,1],
#              'Burst Frequency' : [25, 75, 1],
             'Oscillation Frequency' : [1,20,.25]}


gen_outputs(new_conf_dir, new_output_dir, batch = True, nft_path = 'nftsim/bin/nftsim')