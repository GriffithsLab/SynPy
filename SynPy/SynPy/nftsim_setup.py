import os, sys

def nftsim_setup():
    if not os.path.exists(os.path.join(os.getcwd(), 'nftsim/')):
        create_nftsim = input('NFTsim not found in local directory.  Clone and make? [y/n] ')
        if create_nftsim == 'y':
            os.system('git clone git@github.com:BrainDynamicsUSYD/nftsim.git')
            os.chdir('nftsim/')

            gcc_load_string = "module load gcc/9.4.0 &&" # Assumes being ran from computer cluster with loadable modules (ie. SciNet Niagara)
            os.system(f'{gcc_load_string} make')
            os.chdir('../')
        elif create_nftsim == 'n':
            pass
        else:
            print(f"'{create_nftsim}' not a valid input.")
            sys.exit()