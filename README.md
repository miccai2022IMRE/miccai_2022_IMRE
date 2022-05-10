# ForwardAdaptation

## This readme contains the istructions to run the provided code

### The packages can be installed using the environment.yml file provided.
    - Make sure you already have conda installed on your machine 
    - After installing conda, use conda env create -f environment.yml
    - Environment with name **magmaenv** should be created now 


### To run training, please make sure you have all the correct paths in the Constants.py
    - Please make sure you have a Model_Weights folder to save trained models, if not please create one 
    - The parameter values in the Constants.py are same as the ones used for the experiments
    - Once the training is complete, the models will be saved in the Model_Weights folder
	- Only a subset of the original training and validation data is provided because of size constraints

### To run optimization 
    - First make sure you have Data/Compare_Points/EGMs, Data/Compare_Points/Val_folder, and Data/Sim_L.mat
    - Then call ECGI_optimize() from ECGI.py - it takes two arguments
    - First argument is log folder and the second argument is model filename
    - Example call ECGI_optimize ('ECGI_logs_sim_data/', 'Model_name'). The model name will be searched in Model_Weights Folder


