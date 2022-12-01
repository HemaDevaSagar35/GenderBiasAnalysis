Categorical bias score
Set Up for external librarires
pip install -r requirements.txt

Evaluation script
Add the pretrained model using the below command and change the name of the bert_config file to config.json as the code is testing using the open source transformers
categorical_score.py --lang en --custom_model_path 'model/path'

References:
https://arxiv.org/pdf/2109.05704.pdf

https://github.com/jaimeenahn/ethnic_bias
