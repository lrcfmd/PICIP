# PICIP
Python code for the running, testing and results analysis of PICIP

Requirements may be installed by running pip install -r requirements.txt:  
emmet==2018.6.7  
matplotlib==3.8.4  
mp_api==0.42.1  
numpy==1.21.5  
pandas==2.2.2  
plotly==5.22.0  
pymatgen==2024.5.1  
scipy==1.14.1  
tqdm==4.66.4

For quick use open main.py and in the main function set desired hyperparameters and uncomment desired function for phase field.
The code is structured as follows: phasefield.py defines a phase field class that stores the representation of the phase space, and performs all calculations on it. main.py contains scripts to run simulations using this class. visualise_square.py deals with plotting of the phase fields and simulating_phase_fractions.py with simulating the model error. results_plotting.py contains scripts for analysis of the data. mataerialproject_in.py is a script to create data files from materials project data for computational phase fields. Those used in the paper are already provided. fielddata.py provides the data from mpds (sadly not automated). 
