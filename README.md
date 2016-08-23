#Python-starter-repo

A starter repo, with python folder structure, code template, and conda environment template

## Getting started

### Repo structure
Code is in `bin/`, with different scripts for part 1, part 2A and part 2B. These scripts can be run separately, but 
require files located throughout this repo. 
 
Documents are in `docs/`. The original codetest instructions are in `codetest_instructions.txt`, and my responses are in
`code_test_responses.md`. The file `modeling_notes.txt` is a general brain dump as I've moved through this code 
challenge, but is not edited in any way. 

### Python Environment
Python code in this repo utilizes packages that are not part of the common library. To make sure you have all of the 
appropriate packages, please install [Anaconda](https://www.continuum.io/downloads), and install the environment 
described in environment.yml (Instructions [here](http://conda.pydata.org/docs/using/envs.html), under *Use 
environment from file*, and *Change environments (activate/deactivate)*). 

### To run code
  
To run the Python code, complete the following:
```bash
# Install anaconda environment
conda env create -f environment.yml 
# Make a note of the environment name (e.g. console will show source activate environment_name)

# Activate environment
source activate environment_name

# Run script
cd bin/
python file_name.py
```


## Contact
Feel free to contact me at 13herger@gmail.com
