'''
Which vars do I need here? 
task - model_id - spec
'''

import csv
import sys, os, random, time, torch
import pandas as pd
from os import walk
import torch.nn.functional as F
from glob import glob
from transformers import TrainingArguments, Trainer

from train_functions import timestamp
from rtpt import RTPT
from transformer.modeling import TinyBertForSequenceClassification
from transformer.tokenization import BertTokenizer
from util import load_hf
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
#from sklearn.model_selection import train_test_split # this is in train_util

# vars = sys.argv[1:]
# print(vars)
# print(type(vars[2]))
# print(len(vars))

# assert(len(vars) == 3), "something's wrong with the parameters here. Check that, please. \n call this script with python train [task] [model_id] [spec], where task, model_od and spec need to be a valid string. So e.g. python train 'IMDB' 'bert-base-uncased' all"

# # now we know that we have the right amount of vars 
# task_in = vars[0]
# assert(task_in in ['IMDB', 'Twitter']), 'task name is not valid'
# model_id_in = vars[1]
# assert(model_id_in in ["bertbase", 'bertlarge', "distbase", "distlarge", "robertabase", "robertalarge", "albertbase", "albertlarge"]), model_id_in + ' is not a valid model_id'
# spec_in = vars[2].split()
# print(spec_in)
task_in = "IMDB"
model_id_in = "albertbase"
spec_in = ["N_pro", "N_weat", "N_all", "mix_pro", "mix_weat", "mix_all", "original"]
print('called rate.py {} {} {}'.format(task_in, model_id_in, spec_in))

#log_path = './res_models/logs/rate_{}_{}_{}.txt'.format(task_in, model_id_in, timestamp()) # tu.
#sys.stdout = open(log_path, 'a')

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

 
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, seq_length=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.seq_length = seq_length
        self.label_id = label_id


########## F U N C T I O N S ##########
### from rate.ipynb

def get_test_examples(data_dir,name):
    """See base class."""
    return _create_examples(
        _read_tsv(os.path.join(data_dir, name)), "test")

def _read_tsv(input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

def _create_examples(lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            
            text_a = line[1]
            #text_b = line[4]
            label = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {'1': 1, '0': 0}

    features = []
    for (ex_index, example) in enumerate(examples):

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        seq_length = len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length


        # if output_mode == "classification":
        if example.label  in label_map:
            label_id = label_map[example.label]
            # elif output_mode == "regression":
            #     label_id = float(example.label)
            # else:
            #     raise KeyError(output_mode)

            if ex_index < 1:
                print('logging here')

            features.append(
                InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id=label_id,
                            seq_length=seq_length))
    return features

def get_labels():
        """See base class. This should be changed according to the task"""
        return [1, 0]

def rate(task="IMDB", model_id="tinybert", spec=spec_in):
    '''
    '''
    df_l = pd.read_pickle('{}_l_test'.format(task))
    
    df_exp = []
    for elem in spec_in: 
        df_exp = df_l[['ID', 'text_{}_M'.format(elem), 'text_{}_F'.format(elem), 'label']]
        df_exp['label'][df_exp['label'] == 'pos'] = 1
        df_exp['label'][df_exp['label'] == 'neg'] = 0
        print('rate experimental data type {}: {} and {}'.format(elem, df_exp.columns[1], df_exp.columns[2]))
        print(df_exp.head()), 'rate(): What type of data should be uses? spec does not fit in any categorie'
        male_df_exp = df_exp[['text_{}_M'.format(elem), 'label']]
        female_df_exp = df_exp[['text_{}_F'.format(elem), 'label']]
        male_df_exp = male_df_exp.rename(columns={'text_{}_M'.format(elem): 'text', 'label':'label'})
        female_df_exp = female_df_exp.rename(columns={'text_{}_F'.format(elem): 'text', 'label':'label'})
        male_df_exp.to_csv('text_{}_M.tsv'.format(elem), sep="\t")
        female_df_exp.to_csv('text_{}_F.tsv'.format(elem), sep="\t")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        path = 'res_models/models/IMDB_tinybert_original'
        tokenizer = BertTokenizer.from_pretrained(path, do_lower_case= True)
        model = TinyBertForSequenceClassification.from_pretrained(path, num_labels=2)
        model.to(device)

        model.eval()

        batch_size = 32
        test_examples = get_test_examples(data_dir=os.getcwd(),name = 'text_{}_M.tsv'.format(elem)) #pwd 
        test_features = convert_examples_to_features(test_examples, get_labels(), 512, tokenizer) #check seq length
        test_data, test_labels = get_tensor_data(test_features)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)


        preds = [] 
        for step, batch in enumerate(tqdm(test_dataloader, desc="Iteration", ascii=True)):
            batch = tuple(t.to(device) for t in batch)

            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch
            # if input_ids.size()[0] != batch_size:
            #     continue
            
            logits, atts, reps = model(input_ids, segment_ids, input_mask, is_student=False)
            predictions = torch.nn.functional.softmax(logits, dim=-1)
            preds.append(predictions.detach().cpu().numpy())
            
    
        raw_pred_m = (np.concatenate(preds, axis=0))

        ###female
        test_examples = get_test_examples(data_dir=os.getcwd(),name = 'text_{}_F.tsv'.format(elem)) #pwd 
        test_features = convert_examples_to_features(test_examples, get_labels(), 512, tokenizer) #check seq length
        test_data, test_labels = get_tensor_data(test_features)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
        preds = [] 
        for step, batch in enumerate(tqdm(test_dataloader, desc="Iteration", ascii=True)):
            batch = tuple(t.to(device) for t in batch)

            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch
            
            logits, atts, reps = model(input_ids, segment_ids, input_mask, is_student=False)
            predictions = torch.nn.functional.softmax(logits, dim=-1)
            preds.append(predictions.detach().cpu().numpy())
    
        raw_pred_f = (np.concatenate(preds, axis=0))
        
        y_soft_m = F.softmax(torch.from_numpy(raw_pred_m), dim=1).tolist()
        y_soft_f = F.softmax(torch.from_numpy(raw_pred_f), dim=1).tolist()
        print('saving in df')
        m_soft = [e[0] for e in y_soft_m]
        f_soft = [e[0] for e in y_soft_f]
        m_soft = m_soft + [0 for _ in range(25000-len(m_soft))]
        f_soft = f_soft + [0 for _ in range(25000-len(f_soft))]
        df_exp['pos_prob_m'] = m_soft
        df_exp['pos_prob_f'] = f_soft
        df_exp['bias'] = df_exp['pos_prob_m']-df_exp['pos_prob_f']
        print(df_exp.head())
        
        df_exp.to_pickle('res_results/rating_{}_{}_{}'.format(task, model_id, elem))
        print('saved in df')
    return df_exp

def get_tensor_data(features):
    # if output_mode == "classification":
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    # elif output_mode == "regression":
    #     all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_label_ids, all_seq_lengths)
    return tensor_data, all_label_ids


############################
#tokenizer, model_foo = load_hf(model_id_in, False) # tu.
# assert(model_foo == None)

specs_all = ["N_pro", "N_weat", "N_all", "mix_pro", "mix_weat", "mix_all", "original"]
if spec_in == ["all"]:
    specs = specs_all
    print('SPECs : rate for all specs')
elif type(spec_in)== list:
    specs = spec_in
    print('SPECs : rate for a subset: ', specs)
elif type(spec_in)==str:
    assert(type(spec_in)==list), "spec is not a list here. This will cause issues later." 
    specs = spec_in
    print('SPECs : rate for only one spec: ' + spec_in)
for spec in specs: 
    assert(spec in specs_all), '{} is no legit specification (spec)'.format(spec)
    
rtpt_train = RTPT(name_initials='SJ', experiment_name=task_in, max_iterations=len(specs)*2)
rtpt_train .start()

output = []

tt = rate()
output.append(tt)
print(output)
    


