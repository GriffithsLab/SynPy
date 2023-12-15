import os
import multiprocessing as mp
import traceback
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
    
    def perm_df(self, load_type = 'parallel', threads = min(40, os.cpu_count())):
        """
        load_type ('parallel' [default], 'serial'): Parallelized or serialized .output file loading into df.
        """
        if load_type == 'parallel':
            df_dict = self.parallel_load(threads)
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

    def parallel_load(self, threads):
        """
        Use multiprocesses to parallelize generating dataframe rows.
        """
        print('Parallelized loading...')
        
        manager = mp.Manager()
        df_dict = manager.dict() # Dictionary that gets shared among the multiprocess

        # Create a process pool with a reduced number of processes
        pool = mp.Pool(processes = threads)

        with tqdm(total=len(self.output_files)) as pbar: # Use tqdm to track the progress of the parallel loading
            def update_pbar(_): # Helper function to update the tqdm progress bar
                pbar.update(1)

#             def apply_row_builder(output_file):
#                 return self.df_row_builder(output_file, df_dict)

            results = []
            for output_file in self.output_files:
                result = pool.apply_async(self.df_row_builder, args=(output_file, df_dict), callback=update_pbar)
                results.append(result)

            # Close the pool and wait for all tasks to complete
            pool.close()
            pool.join()

        # Collect results
        for result in results:
            try:
                result.get()
            except Exception as e:
                traceback.print_exc()
                print()

        return df_dict

#     @staticmethod
#     def apply_row_builder(output_file, df_dict):
#         try:
#             self.df_row_builder(output_file)
#         except Exception as e:
#             traceback.print_exc()

    
    def df_row_builder(self, output_file, df_dict):
        """
        For each process, either loaded in a parallized or serialized manner, construct a dataframe for a .output file permutation.
        """

        try:
            output = dot_output(output_file)

            numerical = output.df(gains=False)
            pre_stim = numerical.loc[:output.stim_onset - 10]
            post_stim = numerical.loc[output.stim_onset + output.stim_duration + 110:]
            
            gains = output.df(gains=True)
            pre_gains = gains.loc[:output.stim_onset - 10].mean()
            post_gains = gains.loc[output.time - 10:output.time].mean()

            row = gains.loc[output.time - 10:output.time].mean() # grab average gain from last 10 seconds (post-stim) simulation
            row.name = output.f_name

        except Exception as e:
            traceback.print_exc()


        try:
            # Broadband
            row['V_AUC_delta'] = PSD_delta(
                pre_stim['pop.e.v'],
                post_stim['pop.e.v'],
                output.sampling_rate,
                target_peak = 'broadband',
            )
            
            # 1/f
            pre_exp = float(PSD(pre_stim['pop.e.v'], output.sampling_rate).fm().aperiodic_params_[-1])
            post_exp = float(PSD(post_stim['pop.e.v'], output.sampling_rate).fm().aperiodic_params_[-1])
            row['exponent_delta'] = (pre_exp - post_exp) / pre_exp
            
            
            # Central Frequency
            row['alpha_CF'] = float(PSD(
                post_stim['pop.e.v'], 
                output.sampling_rate).fm_peak_params(target_peak = [8,13], 
                                                     peak_param = 'CF'))
            
            # Peak power
            row['delta_PW_delta'] = PSD_delta(
                pre_stim['pop.e.v'],
                post_stim['pop.e.v'],
                output.sampling_rate,
                target_peak = [1,4],
                peak_param = 'PW'
            )
            
            row['alpha_PW_delta'] = PSD_delta(
                pre_stim['pop.e.v'],
                post_stim['pop.e.v'],
                output.sampling_rate,
                target_peak = [8,13],
                peak_param = 'PW'
            )
            
            row['beta_PW_delta'] = PSD_delta(
                pre_stim['pop.e.v'],
                post_stim['pop.e.v'],
                output.sampling_rate,
                target_peak = [15,30],
                peak_param = 'PW'
            )
            
            # Nu values
            nu_cols = [c for c in pre_stim.columns if fnmatch.fnmatch(c, 'coupling.*.nu')]
            pre_nu = pre_stim[nu_cols].mean()
            post_nu = post_stim[nu_cols].mean()
            
            nu_delta = abs((pre_nu - post_nu) / pre_nu)
            for idx, nu in nu_delta.iteritems():
                row[f'{idx}_delta'] = nu
            
            # Gain values
            gains_delta = abs((pre_gains - post_gains) / pre_gains)
            for idx, gain in gains_delta.iteritems():
                row[f'{idx}_delta'] = gain


            df_dict[row.name] = row

        except Exception as e:
            traceback.print_exc()

    def _construct_perm_df(self, df_dict):
        """
        Construct perm_df from df_dict.  Adds permutation parameters as df column values and sorts frame.
        """
        
        # Convert the dictionary, whose entries are rows of values, to a DataFrame
        perm_df = pd.DataFrame.from_dict(df_dict, orient='index')
        
        # Extract parameter values contained within perm_df indicies and add each as a column value in the df
        tbs_protocol_params = {idx: protocol_params(idx) for idx in perm_df.index}
        for idx, param in tbs_protocol_params.items():
            for param_name, param_val in param.items():
                perm_df.loc[idx, f'PARAM_{param_name}'] = float(param_val)
                
        perm_df['train_dose'] = tbs_train_dose(pulses_per_burst = perm_df['PARAM_bur'], 
                                               inter_burst_freq = perm_df['PARAM_osc'])
    
        num_params = len(next(iter(tbs_protocol_params.values())))
        
        return perm_df.sort_values(by = list(perm_df.columns[-num_params:]), ascending = [True] * num_params)
