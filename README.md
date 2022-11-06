INTRODUCTION
======== 
Although transformer based models like ELMO, Bert, and OpenGPT have pushed the boundaries of NLP across various language tasks, 
they suffer from significant amount of gender and unintended biases. When these large models are knowledge distilled into smaller 
models they propagate the biases, along with the learnings, they have into the distilled student models. As studied in [1], the 
distilled versions have biases exacerbated in them compared to the corresponding teacher or source model. \
The objective here is to explore bias, for example gender bias, unintended bias, in a smaller knowledge distilled version of Bert model, called TinyBert. 
Understand what contributes to the bias and explore the methods that could potentially help mitigate them.

For most of the experimentation, we adapted the offcial github repo on TinyBERT from here [https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT]

Installation
============
Run command below to install the environment(**using python3**)
```bash
pip install -r TinyBert/requirements.txt
```
Download GloVe embedding from here [https://nlp.stanford.edu/data/glove.6B.zip]

Datasets
===========
MLMA hate-speech dataset is found here [https://github.com/HKUST-KnowComp/MLMA_hate_speech].  hate_speech_zip is the file that needs to be downloaded 

Data Augmentation
===========
To create TinyBert from distillation of Bert, one have to perform 1) General distillation, 2) Data Augmentation and 3) Task specific distillation.
The pipeline to augment data is found TinyBERT folder and the following command runs the pipeline:
```bash
python data_augmentation.py --pretrained_bert_model ${BERT_BASE_DIR}$ \
                            --glove_embs ${GLOVE_EMB}$ \
                            --glue_dir ${GLUE_DIR}$ \  
                            --task_name ${TASK_NAME}$
``` 
Note: Create a folder under glue_dir with name of the classification task at hand.

Fine-Tuning Bert
====================
Bert fine-tuned for a particular task acts as the teacher for task specific distillation of TinyBERT. Pipeline for fine-tuning is in TinyBERT folder and the following command runs the fine-tuning pipeline. Note: we use Bert Base Uncased for our experiments.

```bash
python FT_Bert_Classification.py --data_dir ${TASK DATA FOLDER}$   \
                                 --pre_trained_bert ${BERT BASE UNCASED}$  \
                                 --task_name ${TASK NAME}$ \
                                 --output_dir ${OUTPUT DIRECTORY}$ \
                                 --do_lower_case \
                                 --train_batch_size 64

``` 
