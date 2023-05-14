from .nftsim_generator_FUNCTIONS import *
from utils.nftsim import NF

class dot_output:
    pop_nums = {
            1 : 'e',
            2 : 'i',
            3 : 'r',
            4 : 's',
            5 : 'n',
            6 : 'x'}

    conn_mat = {
            1 : 'ee', # excitatory onto itself
            2 : 'ei', # to excitatory from inhibitory
            3 : 'es', # to excitatory from relay
            4 : 'ex', # to excitatory from tms
            5 : 'ie', # to inhibitory from excitatory
            6 : 'ii', # inhibitory onto itself
            7 : 'is', # to inhibitory from relay
            8 : 're', # to reticular from excitatory
            9 : 'rs', # to reticular from reticular
            10 : 'se', # to relay from excitatory
            11 : 'sr', # to relay from reticular    
            12 : 'sn'} # to relay from noise input
    
    def __init__(self, output_path):
        self.output_path = output_path
        self.f_name = os.path.basename(self.output_path)
        
        self.params = self._output_params()
        self.time = float(param_value('Time', self.params)) # Length of simulation (s)
        self.dt = float(param_value('Deltat', self.params)) # Timestep (s)
        self.sampling_rate = 1 / float(param_value('Interval', self.params)) # Number of written datapoints per second of simulation
        self.write_start = float(param_value('Start', self.params)) # Time in simulation when data begins being written (s)
        
        tms_param_dict = { # .conf keyword : class variable name
        'Onset': 'stim_onset', # Time in the stimulation when TMS begins (s)
        'Duration': 'stim_duration', # Total length of stimulation (s)
        'Amplitude': 'stim_amplitude', # Stimulation amplitude of TMS pulses; BELIEVE THIS IS IN THE UNIT OF microvolts (μV)
        'Bursts': 'stim_pulses_per_burst', # Number of TBS pulses per burst (#)
        'Burst Frequency': 'stim_intra_burst_freq', # Frequency of pulses within a burst (Hz)
        'Oscillation Frequency': 'stim_inter_burst_freq' # Frequency of bursts per second (Hz)
        }

        for tms_param in tms_param_dict.keys(): # If the file has TMS in it, assign its parameters as attributes to the class
            try:
                setattr(self, tms_param_dict[tms_param], float(param_value(tms_param, self.params)))
            except ValueError:
                pass
        
    def _output_params(self):
        """
        Parses and returns the initial parameter segment of the output_path file.
        """
        params = [] # param list; each list value is a .output file line
        with open(self.output_path, 'r') as O: # open .output file
            line = O.readline() # read in initial line
            while not line.startswith('====='): # if the line does not begin with '=====' (which prefaces the data)
                params.append(line) # add the line to the param list
                line = O.readline() # move onto next line in file

        return params

    def df(self, gains=False, normalize=False):
        """
        Given the output_path file, constructs a dataframe object with each column representing a field and an index of time.
        """
        Res = NF(nf_output_file = self.output_path) # NFTsim .output class object

        field_dict = {}
        for field in Res.fields: # for each field key
            split_name = field.split('.') # split the string
            if split_name[0] == 'pop': # if the field is a population (ie. Q, V; not a connection)
                split_name[1] = self.pop_nums[int(split_name[1])] # grab the corresponding population label
            else: # otherwise
                split_name[1] = self.conn_mat[int(split_name[1])] # grab the corresponding connection matrix label
            
            field_dict['.'.join(split_name)] = np.squeeze(Res.data[field])
        
        field_df = pd.DataFrame(field_dict, index = Res.time + self.write_start)

        
        if normalize:
            field_df = (field_df - field_df.min()) / (field_df.max() - field_df.min())
       
    
        if gains:
            alpha = float(param_value('alpha|Dendrite 1', self.params)) # rise rate of post-synaptic potential
            beta = float(param_value('beta|Dendrite 1', self.params)) # decay rate of post-synaptic potential
            Q_max = float(param_value('Qmax', self.params)) # population firing rate
            sigmoid_theta = float(param_value('Sigmoid Theta', self.params)) # mean population firing threshold, relative to rest
            sigma = float(param_value('Sigma|Firing', self.params)) # standard deviation of soma voltage, relative to threshold

            gain_dict = {}
            
            ct_connections = {k: v for k, v in self.conn_mat.items() if v not in {'ex', 'sn'}} # remove the non-synaptic connections
            for conn_num, conn_name in ct_connections.items():
                postsynaptic_pop_name = conn_name[0]  # from each connection, grab corresponding postsynaptic pop #

                gain_ab = field_df[f'coupling.{conn_name}.nu'] * (
                    (Q_max * np.exp(-(field_df[f'pop.{postsynaptic_pop_name}.v'] - sigmoid_theta)/sigma)) / 
                    (sigma * (1 + np.exp(-(field_df[f'pop.{postsynaptic_pop_name}.v'] - sigmoid_theta)/sigma))**2))

                gain_dict[conn_name] = gain_ab

            gain = pd.DataFrame(gain_dict)

            gain['g_ese'] = gain['es'] * gain['se'] #  Gese = GesGse,
            gain['g_esre'] = gain['es'] * gain['sr'] * gain['re'] # Gesre = GesGsrGre
            gain['g_srs'] = gain['sr'] * gain['rs'] #  Gsrs = GsrGrs

            gain['X'] = gain['ee'] / (1 - gain['ei']) # Cortical loop gain
            gain['Y'] = (gain['g_ese'] + gain['g_esre']) / ((1 - gain['g_srs'])*(1 - gain['ei'])) # Cortico-thalamic loop gain
            gain['Z'] = -gain['g_srs'] * ((alpha*beta) / (alpha+beta)**2) # Intrathalamic loop gain

            gain['pop.e.v'] = field_df['pop.e.v']
            gain['propagator.ee.phi'] = field_df['propagator.ee.phi']

            return gain

        return field_df