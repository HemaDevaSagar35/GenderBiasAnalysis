ACKNOWLEDGEMENT
==============
Note: For most of the experimentation, we adapted various official github repos. You can find them in Reference section. That being said below are the scripts that we either wrote from scratch or adapted an existing script and upgraded it.

GenderBiasAnalysis/FT_Bert_Classification.py\
GenderBiasAnalysis/task_distill.py\
GenderBiasAnalysis/bias_analysis.py\
GenderBiasAnalysis/result.ipynb


INTRODUCTION
======== 

There are 2 facets here:
1) Training Bert and TinyBert models on MLMA and IMDB datasets
2) Doing various bias analysis with the models obtained from step 1.

You could Skip step 1 and run scripts responsible for Step 2 using the models we generated from our experiments. You can find the models we generated here [https://drive.google.com/drive/folders/1XmLXSMbYAur1mZfqGJfmUaTQGa8BuX1S?usp=share_link]

But incase you want to run Step 1 and re-generate the models, below are the things you have to do:

HOW TO RUN STEP 1
=================

pre-requisites
==============
1. pip install -r TinyBert/requirements.txt
2. Download GloVe embedding from here [https://nlp.stanford.edu/data/glove.6B.zip]
3. Unzip it in GenderBiasAnalysis/TinyBERT/embeddings folder
4. Download bert-base-uncased folder from [https://drive.google.com/drive/folders/1XmLXSMbYAur1mZfqGJfmUaTQGa8BuX1S?usp=share_link] and place it in GenderBiasAnalysis/TinyBERT/
5. Download tinybert-gkd-model folder from [https://drive.google.com/drive/folders/1XmLXSMbYAur1mZfqGJfmUaTQGa8BuX1S?usp=share_link] and place it in GenderBiasAnalysis/TinyBERT/
6. Download the glue_data folder from [https://drive.google.com/drive/folders/1XmLXSMbYAur1mZfqGJfmUaTQGa8BuX1S?usp=share_link] and place them in GenderBiasAnalysis/data/ 

Training Hate Speech model on MLMA
==================================
1. Fine Tune Bert on MLMA dataset with the following command 
```bash
python GenderBiasAnalysis/TinyBERT/FT_Bert_Classification.py --data_dir GenderBiasAnalysis/data/glue_data/MLMA \
                                       --pre_trained_bert GenderBiasAnalysis/TinyBERT/bert-base-uncased \
                                       --task_name MLMA \
                                       --do_lower_case \
                                       --output_dir GenderBiasAnalysis/TinyBERT/output_models \
                                       --num_train_epochs 30
``` 
2. Do intermediate distillation of TinyBERT on MLMA using the following command
```bash
python GenderBiasAnalysis/TinyBERT/task_distill.py --teacher_model GenderBiasAnalysis/TinyBERT/output_models \
                       --student_model drive/MyDrive/tinybert-gkd-model \
                       --data_dir GenderBiasAnalysis/data/glue_data/MLMA \
                       --task_name MLMA \
                       --output_dir GenderBiasAnalysis/TinyBERT/tiny_temp_model \
                       --max_seq_length 128 \
                       --train_batch_size 32 \
                       --num_train_epochs 20 \
                       --aug_train \
                       --do_lower_case
``` 
3. Do prediction layer distillation of TinyBERT on MLMA using the following command
```bash
python GenderBiasAnalysis/TinyBERT/task_distill.py --pred_distill  \
                       --teacher_model GenderBiasAnalysis/TinyBERT/output_models \
                       --student_model GenderBiasAnalysis/TinyBERT/tiny_temp_model \
                       --data_dir GenderBiasAnalysis/data/glue_data/MLMA \
                       --task_name MLMA \
                       --output_dir GenderBiasAnalysis/TinyBERT/tinybert_model \
                       --aug_train \
                       --do_lower_case \
                       --learning_rate 3e-5  \
                       --num_train_epochs  3  \
                       --max_seq_length 64 \
                       --train_batch_size 32

```

Training Sentiment model on IMDB
==================================
1. Fine tune bert on IMDB dataset using the following comamand

```bash
python GenderBiasAnalysis/TinyBERT/FT_Bert_Classification.py --data_dir GenderBiasAnalysis/data/glue_data/IMDB \
                                     --pre_trained_bert GenderBiasAnalysis/TinyBERT/bert-base-uncased \
                                     --task_name IMDB \
                                     --do_lower_case \
                                     --output_dir GenderBiasAnalysis/TinyBERT/imdb_output_models \
                                     --num_train_epochs 30

``` 

2. Do intermediate distillation of TinyBERT on IMDB using the following command

```bash
python GenderBiasAnalysis/TinyBERT/task_distill.py --teacher_model GenderBiasAnalysis/TinyBERT/imdb_output_models \
                       --student_model GenderBiasAnalysis/TinyBERT/tinybert-gkd-model \
                       --data_dir GenderBiasAnalysis/data/glue_data/IMDB \
                       --task_name IMDB \
                       --output_dir GenderBiasAnalysis/TinyBERT/tiny_temp_imdb_model \
                       --max_seq_length 128 \
                       --train_batch_size 32 \
                       --num_train_epochs 20 \
                       --do_lower_case

``` 

3. Do prediction layer distillation of TinyBERT on IMDB using the following command

```bash
python GenderBiasAnalysis/TinyBERT/task_distill.py --pred_distill  \
                       --teacher_model GenderBiasAnalysis/TinyBERT/imdb_output_models \
                       --student_model GenderBiasAnalysis/TinyBERT/tiny_temp_imdb_model \
                       --data_dir GenderBiasAnalysis/data/glue_data/IMDB \
                       --task_name IMDB \
                       --output_dir GenderBiasAnalysis/TinyBERT/tinybert_imdb_model \
                       --do_lower_case \
                       --learning_rate 3e-5  \
                       --num_train_epochs  3  \
                       --max_seq_length 64 \
                       --train_batch_size 32

```

HOW TO RUN STEP 2
=================

Like mentioned at the start, you can run step 2 either using our models directly or by first running Step 1 and re-generating the models.
If you want to run step 2 directly using our models, please do the following pre-requisites first
1. Download the folder output_models, tinybert_model, imdb_output_models, tinybert_imdb_model from here [https://drive.google.com/drive/folders/1XmLXSMbYAur1mZfqGJfmUaTQGa8BuX1S?usp=share_link] and place them in GenderBiasAnalysis/TinyBERT/
2. Download the glue_data folder from [https://drive.google.com/drive/folders/1XmLXSMbYAur1mZfqGJfmUaTQGa8BuX1S?usp=share_link] and place them in GenderBiasAnalysis/data/
NOTE: The above are required if you want to run Step 2 directly using our models.

Now following are the codes you have to run to get all the results under Step 2. For convinience we are categorzing based on the analysis we did

Unintended Bias
==================
Run the ipython notebook named __analysis.ipynb__. This ipython notebook is self explanatory and does all the analysis that was presented in the report and presentation slides.




REFERENCE
===================
[https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT]

