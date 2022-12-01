ACKNOWLEDGEMENT
==============
Note: For most of the experimentation, we adapted various official github repos. You can find them in Reference section. That being said below are the scripts that we either wrote from scratch or adapted an existing script and upgraded it.

GenderBiasAnalysis/TinyBERT/FT_Bert_Classification.py\
GenderBiasAnalysis/TinyBERT/task_distill.py\
GenderBiasAnalysis/TinyBERT/bias_analysis.py\
GenderBiasAnalysis/TinyBERT/result.ipynb\
GenderBiasAnalysis/imdbtests/res_data/IMDB_data_preparation_script.py\
GenderBiasAnalysis/imdbtests/rate.py\
GenderBiasAnalysis/imdbtests/res_plots/biases.ipynb\
GenderBiasAnalysis/imdbtests/res_plots/tables.ipynb\
GenderBiasAnalysis/TinyBERT/seat_analysis.ipynb\
GenderBiasAnalysis/TinyBERT/seat_analysis.py\
GenderBiasAnalysis/TinyBERT/seat_bert_encoder.ipynb

INTRODUCTION
======== 
Note: We used bert base uncased as our teacher model for experimentation. You can find the github repo of this project at [https://github.com/HemaDevaSagar35/GenderBiasAnalysis]

There are 2 facets here:
1) Training Bert and TinyBert models on MLMA and IMDB datasets
2) Doing various bias analysis with the models obtained from step 1.

You could Skip step 1 and run scripts responsible for Step 2 using the models we generated from our experiments. You can find the models we generated here [https://drive.google.com/drive/folders/1XmLXSMbYAur1mZfqGJfmUaTQGa8BuX1S?usp=share_link]

But incase you want to run Step 1 and re-generate the models, below are the things you have to do:

HOW TO RUN STEP 1
=================

pre-requisites
==============
1. Go to GenderBiasAnalysis/TinyBERT/ and run
```
pip install -r requirements.txt
```
2. Download GloVe embedding from here [https://nlp.stanford.edu/data/glove.6B.zip]
3. Unzip it in GenderBiasAnalysis/TinyBERT/embeddings folder
4. Download bert-base-uncased folder from [https://drive.google.com/drive/folders/1XmLXSMbYAur1mZfqGJfmUaTQGa8BuX1S?usp=share_link] and place it in GenderBiasAnalysis/TinyBERT/
5. Download tinybert-gkd-model folder from [https://drive.google.com/drive/folders/1XmLXSMbYAur1mZfqGJfmUaTQGa8BuX1S?usp=share_link] and place it in GenderBiasAnalysis/TinyBERT/
6. Download the glue_data folder from [https://drive.google.com/drive/folders/1XmLXSMbYAur1mZfqGJfmUaTQGa8BuX1S?usp=share_link] and place them in GenderBiasAnalysis/data/ 

Training Hate Speech model on MLMA
==================================
1. Change the working directory to GenderBiasAnalysis/TinyBERT/
2. Fine Tune Bert on MLMA dataset with the following command 
```bash
python FT_Bert_Classification.py --data_dir ../data/glue_data/MLMA \
                                       --pre_trained_bert bert-base-uncased \
                                       --task_name MLMA \
                                       --do_lower_case \
                                       --output_dir output_models \
                                       --num_train_epochs 30
``` 
3. Do intermediate distillation of TinyBERT on MLMA using the following command
```bash
python task_distill.py --teacher_model output_models \
                       --student_model tinybert-gkd-model \
                       --data_dir ../data/glue_data/MLMA \
                       --task_name MLMA \
                       --output_dir tiny_temp_model \
                       --max_seq_length 128 \
                       --train_batch_size 32 \
                       --num_train_epochs 20 \
                       --aug_train \
                       --do_lower_case
``` 
4. Do prediction layer distillation of TinyBERT on MLMA using the following command
```bash
python task_distill.py --pred_distill  \
                       --teacher_model output_models \
                       --student_model tiny_temp_model \
                       --data_dir ../data/glue_data/MLMA \
                       --task_name MLMA \
                       --output_dir tinybert_model \
                       --aug_train \
                       --do_lower_case \
                       --learning_rate 3e-5  \
                       --num_train_epochs  3  \
                       --max_seq_length 64 \
                       --train_batch_size 32

```

Training Sentiment model on IMDB
==================================
1. Change the working directory to GenderBiasAnalysis/TinyBERT/
2. Fine tune bert on IMDB dataset using the following comamand

```bash
python FT_Bert_Classification.py --data_dir ../data/glue_data/IMDB \
                                     --pre_trained_bert bert-base-uncased \
                                     --task_name IMDB \
                                     --do_lower_case \
                                     --output_dir imdb_output_models \
                                     --num_train_epochs 30

``` 

3. Do intermediate distillation of TinyBERT on IMDB using the following command

```bash
python task_distill.py --teacher_model imdb_output_models \
                       --student_model tinybert-gkd-model \
                       --data_dir ../data/glue_data/IMDB \
                       --task_name IMDB \
                       --output_dir tiny_temp_imdb_model \
                       --max_seq_length 128 \
                       --train_batch_size 32 \
                       --num_train_epochs 20 \
                       --do_lower_case

``` 

4. Do prediction layer distillation of TinyBERT on IMDB using the following command

```bash
python task_distill.py --pred_distill  \
                       --teacher_model imdb_output_models \
                       --student_model tiny_temp_imdb_model \
                       --data_dir ../data/glue_data/IMDB \
                       --task_name IMDB \
                       --output_dir tinybert_imdb_model \
                       --do_lower_case \
                       --learning_rate 3e-5  \
                       --num_train_epochs  3  \
                       --max_seq_length 64 \
                       --train_batch_size 32

```

HOW TO RUN STEP 2
=================

Like mentioned at the start, you can run step 2 either using our models directly or by first running Step 1 and re-generating the models. For convinience we are categorizing based on the analysis we did

Unintended Bias
==================
If you want to run this analysis in step 2 directly using our models, please do the following pre-requisites first
1. Download the folder output_models, tinybert_model, imdb_output_models, tinybert_imdb_model from here [https://drive.google.com/drive/folders/1XmLXSMbYAur1mZfqGJfmUaTQGa8BuX1S?usp=share_link] and place them in GenderBiasAnalysis/TinyBERT/
2. Download the glue_data folder from [https://drive.google.com/drive/folders/1XmLXSMbYAur1mZfqGJfmUaTQGa8BuX1S?usp=share_link] and place them in GenderBiasAnalysis/data/
NOTE: The above are required if you want to run this analysis in Step 2 directly using our models.

Now following are the codes you have to run to get all the results under this analysis in Step 2.

Run the ipython notebook named __GenderBiasAnalysis/TinyBERT/analysis.ipynb__. This ipython notebook is self explanatory and does all the analysis that was presented in the report and presentation slides.

Gender Bias
=================
First, place the model configurations in the "GenderBiasAnalysis/imdbtests/res_models/models" folder https://drive.google.com/drive/folders/1XmLXSMbYAur1mZfqGJfmUaTQGa8BuX1S tinybert_imdb_model and imdb_output_models rename them such that the path to them is 'GenderBiasAnalysis/imdbtests/res_models/models/imdb_bertbase_original' and 'GenderBiasAnalysis/imdbtests/res_models/models/IMDB_tinybert_original'

Change the working directory to GenderBiasAnalysis/
Then run the following command

```bash
python imdbtests/res_data/IMDB_data_preparation_script.py | tee imdbtests/data_prep.txt
```

Then run the following command to get your bias calculations for specs in GenderBiasAnalysis/imdbtests/res_results folder
```bash
python -c 'import imdbtests.rate; rate.rate()'
```
After this , you see the results in GenderBiasAnalysis/imdbtests/res_results folder

use __GenderBiasAnalysis/imdbtests/res_plots/biases.ipynb__ and __GenderBiasAnalysis/imdbtests/res_plots/tables.ipynb__ for consolidating and getting results for biases for model of your choice in table and picture format

SEAT Scoring
===================
Run the 2 steps mentioned in Unintended Bias section. 
Run the ipython notebook named __GenderBiasAnalysis/TinyBERT/seat_analysis.ipynb__. Notebook is self explanatory.
SEAT test, results and plot can be found in SEAT folder

Log Probability Bias Score
=============================
Follow steps 1 and 2 mentioned in Unintended Bias section.
To run log probability bias tests use the following command

First, Change the working directory to GenderBiasAnalysis/

```
python TinyBERT/log_probability_bias_analysis.py 
    --eval Log_Probability_Bias/Corpus_Creation/BEC-Pro/BEC-Pro_EN.tsv 
    --model [MODEL PATH] 
    --out Log_Probability_Bias/results/[SAMPLE NAME].csv
```
MODEL PATH for the 4 models would be TinyBERT/output_models, TinyBERT/imdb_output_models, TinyBERT/tinybert_model, TinyBERT/tinybert_imdb_model. To run for all the four models, replace
MODEL PATH with each model path individually and then run. Also replace [SAMPLE NAME].csv, with the name you want to store the file with.

Sample command: 
``` 
python TinyBERT/log_probability_bias_analysis.py  
    --eval Log_Probability_Bias/Corpus_Creation/BEC-Pro/BEC-Pro_EN.tsv  
    --model TinyBERT/tinybert_imdb_model 
    --out Log_Probability_Bias/results/tinybert_result.csv
```

Results and other resources related to log probability tests can be found in __Log_Probability_Bias__ folder 

Categorical Bias Score
======================
First, change the working directory to GenderBiasAnalysis/categorical_bias_score/
Then do

``` 
pip install -r requirements.txt
```

Evaluation script Add the pretrained model using the below command and change the name of the bert_config file to config.json as the code is testing using the open source transformers

``` 
categorical_score.py --lang en --custom_model_path [MODEL PATH]
```

REFERENCE
===================
https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT \
https://github.com/sciphie/bias-bert \
https://github.com/W4ngatang/sent-bias \
https://github.com/marionbartl/gender-bias-BERT\
https://github.com/jaimeenahn/ethnic_bias
