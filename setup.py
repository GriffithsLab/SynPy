from setuptools import setup, find_packages

# Read the contents of requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='SynPy',
    version='0.1',
    description='Python wrapper for interfacing with the NFTsim toolbox created by BrainDynamicsUSYD.',
    author='Kevin Kadak',
    author_email='kevin.kadak@mail.utoronto.ca',
    url='https://github.com/GriffithsLab/SynPy',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)





