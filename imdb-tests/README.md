# bias-bert

### set up
clone repository and cd into bias-bert  

### hprc
ml purge
          
        
As Pytorch comes with CUDA libraries, we don't need to load CUDA modules.the following two modules are sufficient for PyTorch
ml GCCcore/10.2.0 Python/3.8.6
        
you can save your module list with (dl is an arbitrary name)
module save nlp
      
next time when you login you can simply run
module restore nlp

### venv
    
create a virtual environment (the name dlvenv is arbitrary)
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
`python3 res_data/IMDB_data_preparation_script.py | tee data_prep.txt`  

### train
Train the models with train.py. The script is called with three variables, which are (1) the task (i.e. "IMDB" or "Twitter"), (2) the defined model_id of the pretrained model (find a list of all options below) and (3) the data specification(s) (spec) that are used to train the model(s). Each specification determines a different subset of test and training data and results in one model. 

python3 -c 'import train_pytorch; train_pytorch.py'


### possible variables
specs are `"N_pro"`, `"N_weat"`, `"N_all"`, `"mix_pro"`, `"mix_weat"`, `"mix_all"`, `"original"`;  
model_id can be `"bertbase"`, `"bertlarge"`, `"distbase"`, `"distlarge"`, `"robertabase"`, `"robertalarge"`, `"albertbase"`, `"albertlarge"`,  
which correspond to the pretrained [Hugging Face Models](https://huggingface.co/models) `bert-base-uncased`, `bert-large-uncased`, `distilbert-base-uncased`, `distilbert-large-uncased`, `roberta-base`, `roberta-large`, `albert-base-v2`, `albert-large-v2`.   

### evaluate 
python3 -c 'import evaluate_pytorch; evaluate_pytorch.py'

### get evaluations in a neat format
your res_models has all the results and we can use evaluate.ipynb to get all results in a single table type structure(df)

### where is the model abstraction?
It is at line 479 and 78 in train_pytorch.py
tokenizer, model = u.load_hf(model_id, h_droo)  # configuration=configuration)



### License 
cite paper here.  


### Resources 
- IMDB data  
- Stanford data  

