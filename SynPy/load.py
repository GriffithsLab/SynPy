import os
import multiprocessing as mp
import fnmatch
from tqdm import tqdm
import pandas as pd
from .nftsim_generator_FUNCTIONS import list_files, protocol_params, tbs_train_dose
from .output_handler import dot_output
from .spectra_functions import PSD, PSD_delta
from .param_space import valid_iTBS_protocols

class perm_load:
    def __init__(self, output_dir):
        """
        Load permuatations of .output files given a list of paths.
        
        output_dir: path of directorary containing permutations of .output files.
        """
        self.output_dir = output_dir
        self.output_files = list_files(self.output_dir, full_path=True, extension_filter='.output')
        if len(self.output_files) < 1:
            raise Exception('Directory path contains no files.')

        self.output_files = [op for op in self.output_files]# if os.path.basename(op) in valid_iTBS_protocols()]
        
    def __call__(self, load_type = 'parallel'):
        return self.perm_df(load_type)
    
    def perm_df(self, load_type = 'parallel'):
        """
        load_type ('parallel' [default], 'serial'): Parallelized or serialized .output file loading into df.
        """
        if load_type == 'parallel':
            df_dict = self.parallel_load()
        elif load_type == 'serial':
            df_dict = self.serialized_load()
        else:
            raise Exception('load_type must be either "parallel" or "serial".')

        return self._construct_perm_df(df_dict) # previously called xyz_gains or xyz_gainer in notebooks
        
    def serialized_load(self):
        """
        Iteratively generate dataframe rows.
        """
        print('Serialized loading...')
        df_dict = {}
        for output_file in tqdm(self.output_files):
            df_dict[output_file] = self.df_row_builder(output_file, df_dict)

        return df_dict

    # Function to parallelize the loading process
    def parallel_load(self):
        """
        Use multiprocesses to parallelize generating dataframe rows.
        """
        print('Parallelized loading...')
        manager = mp.Manager()
        df_dict = manager.dict() # Dictionary that gets shared among the multiprocess

        # Create a process pool with the number of available CPU cores
        pool = mp.Pool() # 'processes' argument may be passed if loading hits memory limit

        with tqdm(total=len(self.output_files)) as pbar: # Use tqdm to track the progress of the parallel loading
            def update_pbar(_): # Helper function to update the tqdm progress bar
                pbar.update(1)

            # Apply the paralloader function to each output file in parallel
            for output_file in self.output_files:
                pool.apply_async(self.df_row_builder, args=(output_file, df_dict), callback=update_pbar)

            # Close the pool
            pool.close()
            pool.join()

        return df_dict
    
#     @staticmethod
    def df_row_builder(self, output_file, df_dict):
        """
        For each process, either loaded in a parallized or serialized manner, construct a dataframe for a .output file permutation.
        """

        try:
            output = dot_output(output_file)

            numerical = output.df(gains=False)
            gains = output.df(gains=True)

            row = gains.loc[output.time - 10:output.time].mean() # grab average gain from last 10 seconds (post-stim) simulation
            row.name = output.f_name

            pre_stim = numerical.loc[:output.stim_onset - 10]
            post_stim = numerical.loc[output.stim_onset + output.stim_duration + 110:]

        except Exception as e:
            print(e)  # Print the exception message


        try:

            row['V_AUC_delta'] = PSD_delta(
                pre_stim['pop.e.v'],
                post_stim['pop.e.v'],
                output.sampling_rate,
                target_peak = 'broadband',
            )

            row['alpha_CF'] = float(PSD(
                pre_stim['pop.e.v'], 
                output.sampling_rate).fm_peak_params(target_peak = 10, 
                                                     peak_param = 'CF')
           )
            
            row['FOOOF_PW_delta'] = PSD_delta(
                pre_stim['pop.e.v'],
                post_stim['pop.e.v'],
                output.sampling_rate,
                target_peak = 10,
                peak_param = 'PW'
            )

            nu_cols = [c for c in pre_stim.columns if fnmatch.fnmatch(c, 'coupling.*.nu')]
            pre_nu = pre_stim[nu_cols].mean()
            post_nu = post_stim[nu_cols].mean()

            nu_delta = (post_nu - pre_nu) / pre_nu
            for idx, nu in nu_delta.iteritems():
                row[f'delta_{idx}'] = nu
            
            gains_delta = abs((post_gains - pre_gains) / pre_gains)
            for idx, gain in gains_delta.iteritems():
                row[f'delta_{idx}'] = gain

#             xyz_cols = [c for c in pre_stim.columns if c in ('X', 'Y', 'Z')]
#             pre_xyz
#             post_xyz


    #         row['nu_ts'] = numerical_normalize[cols][output.stim_onset:output.stim_onset + output.stim_duration]

    #         itbs = numerical['pop.x.q'][int(output.stim_onset):int(output.stim_onset + output.stim_duration)]
    #         row['itbs'] = itbs.values
    #         row['itbs_psd'] = sp.PSD(itbs, output.sampling_rate).values


            df_dict[row.name] = row

        except Exception as e:
            print(e)  # Print the exception message

    def _construct_perm_df(self, df_dict):
        """
        Construct perm_df from df_dict.  Adds permutation parameters as df column values and sorts frame.
        """
        
        # Convert the dictionary, whose entries are rows of values, to a DataFrame
        perm_df = pd.DataFrame.from_dict(df_dict, orient='index')
        
        # Extract parameter values contained within perm_df indicies and add each as a column value in the df
        protocol_idx = {i: protocol_params(i) for i in perm_df.index}
        for idx, param in protocol_idx.items():
            for param_name, param_val in param.items():
                perm_df[idx, f'PARAM_{param_name}'] = float(param_val)
                
        perm_df['train_dose'] = tbs_train_dose(pulses_per_burst = perm_df['PARAM_bur'], 
                                               inter_burst_freq = perm_df['PARAM_osc'])
    
        num_params = len(next(iter(protocol_idx.values())))
        return perm_df.sort_values(by = list(perm_df.columns[-num_params:]), ascending = [True] * num_params)
