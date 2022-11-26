from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, recall_score, precision_score, confusion_matrix, roc_auc_score

from transformer.modeling import BertForSequenceClassification, TinyBertForSequenceClassification
from transformer.tokenization import BertTokenizer
from transformer.optimization import BertAdam
from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('debug_layer_loss.log')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()

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


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class UBDSProcessor(DataProcessor):
    """Processor for any binary classification "given Task" data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
    
    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class. This should be changed according to the task"""
        return ["non-toxic", "toxic"]

    def _create_examples(self, lines, set_type):
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

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

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
        label_id = label_map[example.label]
        # elif output_mode == "regression":
        #     label_id = float(example.label)
        # else:
        #     raise KeyError(output_mode)

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {}".format(example.label))
            logger.info("label_id: {}".format(label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          seq_length=seq_length))
    return features

def do_eval(model, task_name, eval_dataloader,
            device, eval_labels, num_labels):
    eval_loss = 0
    nb_eval_steps = 0
    preds = []

    for batch_ in tqdm(eval_dataloader, desc="Evaluating"):
        batch_ = tuple(t.to(device) for t in batch_)
        with torch.no_grad():
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_

            logits, _, _ = model(input_ids, segment_ids, input_mask)

        loss_fct = CrossEntropyLoss(weight = torch.tensor([8.3, 1.]).to(device))
        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
   
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    preds = preds[0]
    # if output_mode == "classification":
    preds = np.argmax(preds, axis=1)
    
    result = compute_metrics(task_name, preds, eval_labels.numpy())
    result['eval_loss'] = eval_loss

    return result

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

def predictions(model, task_name, data_dir, max_seq_length, do_lower_case=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    
    processor = UBDSProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(model, do_lower_case= do_lower_case)
    model = TinyBertForSequenceClassification.from_pretrained(model, num_labels=num_labels)
    model.to(device)

    model.eval()

    batch_size = 32 
    test_examples = processor.get_test_examples(data_dir)
    test_features = convert_examples_to_features(test_examples, label_list, max_seq_length, tokenizer)
    test_data, test_labels = get_tensor_data(test_features)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)


    preds = [] 
    for step, batch in enumerate(tqdm(test_dataloader, desc="Iteration", ascii=True)):
        batch = tuple(t.to(device) for t in batch)

        input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch
        # if input_ids.size()[0] != batch_size:
        #     continue
        logits, atts, reps = model(input_ids, segment_ids, input_mask, is_student=True)
        predictions = torch.nn.functional.softmax(logits, dim=-1)
        preds.append(predictions.detach().cpu().numpy())
    
    return np.concatenate(preds, axis=0)

def read_data(data_dir):
    file_path = os.path.join(data_dir, 'test.tsv')
    data = pd.read_csv(file_path, sep='\t')
    data['label'] = 0
    data.loc[data['toxicity'] == 'toxic', 'label'] = 1
    return data

def metrics_analysis(predictions, data_dir):
    # ref for this words are coming from https://dl.acm.org/doi/pdf/10.1145/3278721.3278729
    
    identity_word = ['queer', 'bisexual', 'gay', 'lesbian', 'japanese', 'homosexual', 'chinese',
                     'transgender', 'buddhist', 'middle eastern', 'canadian', 'european', 'asian',
                     'sikh', 'american', 'teenage', 'african american', 'african', 'indian', 'older',
                     'millenial', 'female', 'latinx', 'elderly', 'nonbinary', 'male', 'mexican', 'hispanic',
                     'old', 'younger', 'middle aged', 'heterosexual', 'muslim', 'taoist', 'blind', 'latino',
                     'protestant', 'lgbtq', 'paralyzed', 'white', 'latina', 'jewish', 'christian', 'black',
                     'lgbt', 'deaf', 'catholic', 'straight', 'trans', 'young']
    
    identity_word =  list(set(identity_word))
    
    identity_wise_split = {}
    data = read_data(data_dir)
    data['index'] = list(range(data.shape[0]))
    auc_dict = {}

    for word in tqdm(identity_word):
        sent = []
        for i, [sentence, lab] in enumerate(data[['phrase', 'label']].values.tolist()):
            if word.lower() in sentence.lower():
                sent.append(i)
        print(len(sent))
        identity_wise_split[word] = sent.copy()
    
    for word in tqdm(identity_word):
        ### AUC - subgroup
        
        indexes = identity_wise_split[word]

        print(word)
        print(data.iloc[indexes[0], 1])

        y_true = data.iloc[indexes, 3]
        y_score = predictions[indexes, 1]
        score_1 = roc_auc_score(y_true, y_score)
        
        ### AUC - background positive subgroup negative
        positive_index = data.loc[data['label'] == 1, 'index'].values.tolist()
        intersection = set(positive_index).intersection(set(indexes))
        positive_background_index = list(set(positive_index) - intersection)
        indexes_neg = list(set(indexes) - intersection)
        final_indexes = positive_background_index + indexes_neg
        print("positive background index {}".format(len(positive_background_index)))
        print("sum of positive background is {}".format(data.iloc[positive_background_index, 3].sum()))
        print("indexes negetives {}".format(len(indexes_neg)))
        print("sum indexes negetives {}".format(data.iloc[indexes_neg, 3].sum()))
        

        y_true = data.iloc[final_indexes, 3]
        y_score = predictions[final_indexes, 1]
        score_2 = roc_auc_score(y_true, y_score)

        ### AUC - Background Negative Subgroup Positive
        negative_index = data.loc[data['label'] == 0, 'index'].values.tolist()
        intersection = set(negative_index).intersection(indexes)
        negative_background_index = list(set(negative_index) - intersection)
        indexes_pos = list(set(indexes) - intersection)

        final_indexes = negative_background_index + indexes_pos

        print("negative_background_index {}".format(len(negative_background_index)))
        print("sum of negative_background_index is {}".format(data.iloc[negative_background_index, 3].sum()))
        print("indexes positives {}".format(len(indexes_pos)))
        print("sum indexes positives {}".format(data.iloc[indexes_pos, 3].sum()))


        y_true = data.iloc[final_indexes, 3]
        y_score = predictions[final_indexes, 1]
        score_3 = roc_auc_score(y_true, y_score)

        auc_dict[word] = [score_1, score_2, score_3]

    overall = roc_auc_score(data['label'].values, predictions[:,1])


    return auc_dict, overall
    

    






    




