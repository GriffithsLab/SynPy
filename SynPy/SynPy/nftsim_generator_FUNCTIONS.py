# Data & directory navigation
import os,time,sys,glob,numpy as np,pandas as pd
import re
import subprocess
from tqdm import tqdm # gives progress status of for loop processing
from IPython import get_ipython

import statistics
import math
from itertools import product
from scipy.signal import welch
from utils.nftsim import NF
from fooof import FOOOF

# Visualization
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import seaborn as sns

# sys.path.append('nftsim/')


def keyword_instances(keyword, conf_txt): # returns all present instances of a keyword in conf_txt (index & line content)
    """
    Given a keyword and list of text lines (conf_txt), returns all instances of the kw and the lines numbers in which they are present.
    """
    
    instances = {}
    for line_idx, line_contents in enumerate(conf_txt):
        kw_match = re.search(keyword + "(.+)", line_contents, re.IGNORECASE)
        if kw_match:
            instances[line_idx] = line_contents

    return instances


def param_value(kw, conf_txt, _include_idxs_ = False):
    """
    Arguments:
        Keyword (string)
        'keyword|keyword_filter'

    Given a keyword and a sometimes necessary keyword filter, returns the corresponding parameter value of the keyword.  The keyword
    filter isolates instances of the keyword to only the line(s) also containing the keyword filter.
    
    Returns:
        Corresponding parameter value
    """
    if '|' in kw:
        keyword, keyword_filter = kw.split('|')
    else:
        keyword = kw; keyword_filter = ''
    
    
    keyword = keyword.strip(':'); keyword_filter = keyword_filter.strip(':') # remove ';' from passed keyword strings
    kw_instances = keyword_instances(keyword, conf_txt)
    
    if not kw_instances: # If there are no instances of the keyword in the conf_txt, return None
        return None
    elif len(kw_instances) > 1 and keyword_filter == '':
        for l, c in kw_instances.items():
            print(f'{l, c}\n')
        keyword_filter = input(f'{len(kw_instances)} instances of {keyword} and no filter keyword passed.  Enter keyword filter: ')
    
    
    for line_idx, line_contents, in kw_instances.items(): # if the kw_filter is not also in the line, iterate to the next line
        if keyword_filter != '' and not re.search(keyword_filter + "(:.+)", line_contents, re.IGNORECASE):
            continue

        # adds a space after ':' in any instances there is none, removes newline tag, and splits the conf_txt on spaces
        split_loi = re.sub(r'(?<=:)(?! )', ' ', conf_txt[line_idx]).strip('\n').split(" ")

        for value_idx, value in enumerate(split_loi): # Iterate through the line objects until the keyword object is found
            if re.match(keyword, value, re.IGNORECASE):
                for vplus_idx, vplus in enumerate(split_loi[value_idx:]):
                    try:
                        float(vplus) # try converting the value at index vplus_idx after the index of the found kw to a float
                        if _include_idxs_:
                            return line_idx, value_idx + vplus_idx, vplus # split loi index and value of corresponding kw
                        else:
                            return vplus # by default, only return the corresponding keyword (parameter) value
                    except:
                        continue # otherwise, jump to the next object in the split loi

    return None # If all lines and strings have been iterated through without kw-kw_filter matches, return None        


def _replace_value_(kw, new_value, conf_txt):
    """
    Given a keyword and keyword filter (used for multiple instances of the keyword in the conf_txt), replaces the corresponding 
    parameter (keyword) value with a new_value in the passed conf_txt object. 
    """
    new_value = str(new_value)
    if '|' in kw:
        keyword, keyword_filter = kw.split('|')
    else:
        keyword = kw; keyword_filter = ''
    
    param_idxs_value = param_value(kw, conf_txt, _include_idxs_ = True)
    if param_idxs_value == None:
        raise Exception(f"No instances of '{keyword}' on lines containing '{keyword_filter}' to replace.")
    
    line_num, value_idx, value = param_idxs_value 
    
    split_loi = re.sub(r'(?<=:)(?! )', ' ', conf_txt[line_num]).split(" ")
    if split_loi[value_idx] == split_loi[-1]: # if the kw is found at the last index (ie. end of the line)
        new_value += ('\n') # add a new line tag 
    
    split_loi[value_idx] = new_value # update the old value with the new at split line index vv
    conf_txt[line_num] = " ".join(split_loi) # re-combine the objects in split loi into a single, updated loi
    
    print(f"Replacing {kw} '{value}' with '{new_value}' -- {conf_txt[line_num]}")
    conf_txt[:] = conf_txt # Update the conf_txt object in memory with the modified line
    
    
def update_param(keyword, new_value, conf_txt):
    _replace_value_(keyword, new_value, conf_txt)


def save_confs(new_confs, conf_dir = os.path.join(os.getcwd(), 'confs/')):
    """
    Takes a dictionary object containing a conf_string [key] and conf_content / conf_txt [value]
    
    Generate a new .conf file for each item in the passed new_confs dict
    """
    
    if not os.path.exists(conf_dir): # if the directory doesn't exist, create it
        os.makedirs(conf_dir)
    
    for conf_name, conf_contents in new_confs.items():
        open(os.path.join(conf_dir, conf_name), 'w+').writelines(conf_contents)

    print()
    print(f'Wrote {len(new_confs)} new conf files to: {conf_dir}')

    
def generate_permutations(loop_dict):
    """
    Takes a dictionary object with the name of the parameter being manipulated as the key, and the [start, stop, stepsize] list
    as the corresponding value.
    Ex.
    {'ppb': [1, 20, .25],
     'osc': [1, 20, .25],
     'amp': [70, 120, 5]}
     
     
    Returns a list containing tuple objects of each unique permutation.
    Ex.
    [(1, 1.0, 70),
     (1, 1.0, 75),
     (1, 1.0, 80),
     ...]
    """
    
    # Generate a list of loop ranges
    loop_ranges = [np.arange(start, stop + step, step) for loop_name, [start, stop, step] in loop_dict.items()]

    # Use the product function from the itertools module
    # to generate all permutations of the loop ranges
    perm_list = product(*loop_ranges)

    # Capture each unique permutation in a list
    all_permutations =  [perm for perm in perm_list]

    # Return all unique permutations
    return all_permutations

def quick_conf(f, conf_dir, output_dir, nft_path = 'nftsim/bin/nftsim', params = {}, gains = False, full_path_name = False, save_csv = False):
    """
    Given a .conf file, generates a .output file and returns its contents as a dataframe.
    """
    
    with open(f, 'r') as F:
        conf_txt = F.readlines()
        
    gen_file_string = str(f.split('.')[0]) + '_QUICK' # take the f name string, remove the '.conf', append tag
    conf_name = gen_file_string + '.conf'
    output_name = gen_file_string + '.output'
    
    conf_path = os.path.join(conf_dir, conf_name)
    output_path = os.path.join(output_dir, output_name)
    
    # Update the value of each parameter within the passed 'params' dictionary, if the dict contains items
    if params:
        [update_param(kw, value, conf_txt) for kw, value in params.items()]
    
    save_confs(
       new_confs = {conf_name:conf_txt[:]},
       conf_dir = conf_dir)
    
    
    # Generate .output file from .conf file
    nftsim_shell_code = f'{nft_path} -i {conf_path} -o {output_path}'
    # If the code is executed from a jupyter notebook, load the gcc module in the shell instance when calling nftsim
    if get_ipython() is not None: 
        nftsim_shell_code = "module load gcc/9.4.0 && " + nftsim_shell_code
    subprocess.run(nftsim_shell_code, shell=True, capture_output = True)

    df = output_to_df(output_path, gains = gains, full_path_name = full_path_name)

    if save_csv: 
        if not os.path.exists(os.getcwd() + '/csv/'): os.makedirs(os.getcwd() + '/csv/')
        df.to_csv(f"/{os.getcwd()}/csv/{f.split('.')[0]}.csv")
    
    print()
    print(df)
    
    return df

def gen_outputs(conf_dir,
                output_dir,
                batch,
                nft_path): # run a grid job by default.  if passed as true, instead submit jobs to terminal one by one


    pause_length = 5

    job_submit_file_str_tpl = \
    """#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --output <BATCH_STATUS_FNAME>
#SBATCH --open-mode append
#SBATCH --time=1:00:00

module load python/3.8.5
module load gcc/9.4.0

cd <OUTDIR>

<NFT_PATH> -i <CONF_FNAME> -o <OUT_FNAME>

    """
    
    conf_files = list_files(conf_dir, full_path = True, extension_filter = '.conf')
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    for conf_file in conf_files:
        output_file = conf_file.replace(conf_dir, output_dir).replace('.conf', '.output')
        
        if batch: 

            job_submit_file = conf_file.replace('.conf','_job_submit.sh')
            job_status_file = conf_file.replace('.conf', '_job_status.out')

            job_submit_file_str = job_submit_file_str_tpl\
                                .replace('<BATCH_STATUS_FNAME>', os.path.join(output_dir, job_status_file))\
                                .replace('<OUTDIR>', output_dir)\
                                .replace('<NFT_PATH>', nft_path)\
                                .replace('<CONF_FNAME>', conf_file)\
                                .replace('<OUT_FNAME>', output_file)

            open(job_submit_file, 'w+').write(job_submit_file_str)

            cmdstr = f'sbatch {job_submit_file}'
            print(cmdstr)

            os.system(cmdstr)

        elif batch == False:
            gcc_load_string = "module load gcc/9.4.0 &&"
            os.system(f'{gcc_load_string} {nft_path} -i {conf_file} -o {output_file}')
            print(f'Output file: {output_file.split("/")[-1]} complete')
            print('===============')


def _natural_sort_(string_list):
    """
    Given a list of strings, returns them in natural alphanumeric order.
    """
    
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    
    return sorted(string_list, key=alphanum_key)    


def list_files(file_dir, full_path = False, extension_filter = ''):
    """
    Given a directory containing multiple files, returns a sorted alphanumeric list of their name strings.
    
    Optional parameter: 
        full_path (default: False) may be set to True to return the absolute path string instead of only the file name.
    """
    if full_path:
        # returns list of absolute path strings
        return _natural_sort_([o for o in glob.glob(os.path.join(file_dir, '*')) if o.endswith(extension_filter)]) 
    else:
        # returns list of file name strings only
        return _natural_sort_([o for o in os.listdir(file_dir) if o.endswith(extension_filter)])
        

            
def firing_lock_check(time_s, conf_txt, stim_onset, stim_duration, detect_threshold = .90):
    print(f'time_s: {time_s}')
    
    qmax = float(param_value('Qmax:', conf_txt)[1]) # Maximum firing rate allowed in the simulation
#     stim_start_time =  float(param_value('Onset:', conf_txt, instance_guide = tms_type)[1]) # Time into simulation when stimulation begins being administered
    stim_start_time = stim_onset
#     stim_duration = float(param_value('Duration:', conf_txt)[1]) # Total time in seconds stimulation is administered
    stim_duration = stim_duration
    on_time = spike_length = int(param_value('On:', conf_txt)[1]) # Train interval; how long in seconds a spike occurs for; almost always proportional to ON time
    off_time = int(param_value('Off:', conf_txt)[1]) # Inter-train interval
    train_length = on_time + off_time # total time (s) of a stimulation ON + OFF cycle (1 full train)
    num_spikes = int(stim_duration / train_length) # duration of stimulation / the stimulation incremenet in seconds
    sampling_freq = 1 / float(param_value('Interval:', conf_txt)[1]) # in Hz
    
    
    time_s = time_s.loc[stim_start_time : (stim_start_time + stim_duration)] # Trim the timeseries down to only the TMS ON portion
    for spike in range(num_spikes): # for each spike in the total number of spikes that occur in the stim. time series:
        spike_timing = (spike * train_length) # time (in seconds) in which a given spike first occurs
        spike_idx = int(spike_timing * sampling_freq)  # positional index of when a spike first occurs
        

        spike_firing_rate = time_s.iloc[spike_idx].values
        bt_spike_firing_rate = min(time_s.iloc[int((spike_timing + spike_length) * sampling_freq) : int((spike_timing + train_length) * sampling_freq)].values)
        
        if (detect_threshold * spike_firing_rate) < bt_spike_firing_rate and bt_spike_firing_rate > (detect_threshold * qmax):
            return {spike: int((sampling_freq * stim_start_time) + (spike_idx - sampling_freq))}
    
    return None


def gen_ts_grid(f, conf_dir, output_dir, grid_type = 'ts'):
    
    with open(f, 'r') as F: # read in template conf for param_value data
        conf_txt = F.readlines()
        
    con_num = param_value('Coupling:', conf_txt)[1]
    
    ppb_size = 8
    osc_size = 10

    fig, ax = plt.subplots(figsize = (50, 30), 
                           ncols = ppb_size, 
                           nrows = osc_size)
    
    sampling_freq =  1 / float(param_value('Interval:', conf_txt)[1])
    stim_onset = float(param_value('Onset:', conf_txt, instance_guide = 'TBS')[1])
    stim_duration = float(param_value('Duration:', conf_txt)[1])
    on_time = float(param_value('On:', conf_txt)[1])
    off_time = float(param_value('Off:', conf_txt)[1])
    
    if grid_type == 'ts': grid_fields = [f'coupling.{con_num}.nutilde', f'coupling.{con_num}.nu']
    elif grid_type == 'psd': grid_fields = [f'pop.{con_num}.v']

    output_dicts = read_outputs(f, conf_dir, output_dir, grid_fields)

    for output_name, output_dict in output_dicts.items(): # for output name (should be 80 objects)
        ts_df = pd.DataFrame()
        for field, ts in output_dict.items():
            ts_df[field] = ts.iloc[:, 0]

        print(ts_df)
        ppb = output_name.split('ppb')[1].split('_')[0]
        osc = output_name.split('osc')[1].split('_')[0]
        print()
        print(f'ppb: {ppb}')
        print(f'osc: {osc}')

        ax_item = ax[(osc_size - 1) - (int(osc) - 1), int(ppb) - 1]


        if grid_type == 'ts':
            ax_item.plot(ts_df)

            num_pulses_given = 600
            time600 = round(num_pulses_given*(on_time+off_time)/(int(osc)*int(ppb)*on_time), 2)
            print(time600)
            index_to_look = int(time600 + int(stim_onset))
            ax_item.axvline(index_to_look, linestyle = '--', c = 'purple')


            start_val = float(ts_df[f'coupling.{con_num}.nutilde'].iloc[0])

            if index_to_look > float(ts_df[f'coupling.{con_num}.nutilde'].index[-1]): end_val = None
            else: end_val = float(ts_df[f'coupling.{con_num}.nutilde'].loc[index_to_look])               


            if end_val == None:
                diff = 'Out of range'
            else: 
                diff = f'{(end_val - start_val) / start_val:.2%}'

            ax_item.text(600, start_val, diff)
            print(diff)
            
        elif grid_type == 'psd':
            ## WHEN forwarding a ts for psd plotting, make sure the specific column is grabbed, not the entire df that contains 1 column
            voltage = ts_df[f'pop.{con_num}.v']

            ts_pre_stim = voltage[:int(stim_onset)]
            ts_post_stim = voltage[int(stim_onset) + int(stim_duration) + 5:]


            for window, clr in zip([ts_pre_stim, ts_post_stim], ['dodgerblue', 'blueviolet']):
                valu = np.array(window)
                freqs, mypsd = welch(valu, fs = sampling_freq)
                
                ax_item.plot(freqs, mypsd, color = clr)

            ax_item.set_xscale('log')
            ax_item.set_yscale('log')
                
        ax_item.set_title(f'Pulses/Burst: {ppb}, Theta: {osc} Hz')

    plt.tight_layout()

    if not os.path.exists(os.getcwd() + '/grids/'):
        os.makedirs(os.getcwd() + '/grids/')

    plt.savefig(os.getcwd() + '/grids/' + f.split('.')[0] + f'_{grid_type}_grid.png')


def pre_post_PSD(pre_signal, post_signal, sampling_freq):
    """
    pre_signal
    post_signal
    
    sampling_freq
    
    returns:
        pre-post PSD dataframe, with index values representing frequency bins and column values representing signal power within each
    """

    pre_freqs, pre_power = welch(pre_signal.values.reshape(len(pre_signal)), fs = sampling_freq)
    post_freqs, post_power = welch(post_signal.values.reshape(len(post_signal)), fs = sampling_freq)
    
    welch_df = pd.DataFrame({'pre_power': pre_power, 
                             'post_power' : post_power}, index = pre_freqs)
    welch_df.index.names = ['frequency_bins']
    
    return welch_df
    

def AUC_power_delta(pre_signal,
                    post_signal,
                    sampling_freq, 
                    
                    fooof_correct = False):
    """
    eirs_signal: Takes a pandas dataframe/series containing a series to compare pre-post AUC power (ex. pop.1.v, propagator.1.phi)
    """
    
    welch_df = pre_post_PSD(pre_signal, post_signal, sampling_freq)
    
    # PSD with FOOOF 1/f correction
    if fooof_correct:
        bins = np.array(welch_df.index)

        pre_spectrum = welch_df[['pre_power']].values.flatten()
        post_spectrum = welch_df[['post_power']].values.flatten()

        freq_range = [bins.min(), bins.max()]

        fm = FOOOF()

        fm.fit(bins, pre_spectrum, freq_range)
        pre_fooof = fm.get_params('peak_params')

        fm.fit(bins, post_spectrum, freq_range)
        post_fooof = fm.get_params('peak_params')

        alpha_freq = 10
        pre_pwr = min(pre_fooof, key = lambda x: min(abs(i-alpha_freq) for i in x))[2]
        post_pwr = min(post_fooof, key = lambda x: min(abs(i-alpha_freq) for i in x))[2]

        FOOOF_AUC_diff = (pre_pwr-post_pwr)/(pre_pwr)
        
        return FOOOF_AUC_diff
    
    
    pre_stim_alpha_bins = welch_df['pre_power'][8:13]
    post_stim_alpha_bins = welch_df['post_power'][8:13]
    
    pre_alpha_auc = np.trapz(pre_stim_alpha_bins)
    post_alpha_auc = np.trapz(post_stim_alpha_bins)
    
    AUC_perc_delta = (pre_alpha_auc-post_alpha_auc)/pre_alpha_auc
    
    return AUC_perc_delta


def graph_PSD_delta(pre_signal, 
                    post_signal,
                    sampling_freq):
    """
    Argument:
        eirs_grid multi-index dataframe
    
    Plots interact_manual PSD graph for jupyter notebook
    """
    welch_df = pre_post_PSD(pre_signal, post_signal, sampling_freq)

    psd_fig, psd_ax = plt.subplots(figsize = (20,10))

    welch_df['pre_power'].plot(ax=psd_ax, logx = False, logy=True, c='black', label = 'Pre-stim Voltage', linewidth = 4, linestyle='--')
    welch_df['post_power'].plot(ax=psd_ax,logx = False, logy=True, c='indigo', label = 'Post-stim Voltage', linewidth = 4)

    psd_ax.axvspan(8, 
        13, 
        color = 'pink',
        alpha = .5) 


    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)

    psd_ax.set_ylabel('Spectral Power [$V^2$/Hz]', fontsize = 30)
    psd_ax.set_xlabel('Frequencies [Hz]', fontsize = 30)
#     psd_ax.set_title(f'Pre vs. Post Stimulation Population Voltage PSD | bur={ppb}, osc={osc}', fontsize = 30)
    psd_ax.tick_params(axis='both', which='major', labelsize=20)
    psd_ax.set_xscale('log', base = 10)

    psd_ax.legend()
    plt.show()