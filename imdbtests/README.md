# bias-bert

Majority of code is written following https://github.com/sciphie/bias-bert which is the repository for the referred paper

### set up
clone repository and cd into bias-bert  

### do this only if you use hprc
ml purge
          
        
As Pytorch comes with CUDA libraries, we don't need to load CUDA modules.the following two modules are sufficient for PyTorch
ml GCCcore/10.2.0 Python/3.8.6
        
you can save your module list with (dl is an arbitrary name)
module save nlp
      
next time when you login you can simply run
module restore nlp

### venv
    
create a virtual environment (the name env is arbitrary)
cd $SCRATCH (wherever your proj is)

first upgrade your pip3 version
pip3 install --upgrade pip

pip3 install virtualenv
virtualenv env
source env/bin/activate

then, install requirements by running below command
pip3 install -r requirements.txt
install any missing requirements as required if you get errors.


export PYTHONPATH="${PYTHONPATH}:/scratch/user/sreeja_govardhana/nlp-project" (change path accordingly)

export TRANSFORMERS_CACHE=/scratch/user/sreeja_govardhana/nlp-project/hgcache

export MPLCONFIGDIR=/scratch/user/sreeja_govardhana/nlp-project/mpcache

### get and prepare data 
`python3 imdbtests/res_data/IMDB_data_preparation_script.py | tee data_prep.txt`  

Check after this step if you have
IMDB_l_test file in your folders

Place the model configurations in the "res_models/models" folder
https://drive.google.com/drive/folders/1XmLXSMbYAur1mZfqGJfmUaTQGa8BuX1S 
tinybert_imdb_model
and 
imdb_output_models
rename them such that the path to them is
'res_models/models/imdb_bertbase_original'
and
'res_models/models/IMDB_tinybert_original'

 (some files too big to upload hence get them from drive )
  

### rate for bias measurement
run below command
python3 -c 'import imdbtests.rate; rate.rate()'
to get your bias calculations for specs in res_restults folder

After this , you see the results in res_results folder

use biases.ipynb and tables.ipynb for consolidating and getting results for biases for model of your choice in table and picture format

