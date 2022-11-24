### This is a script for fine-tuning bert for classification

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, recall_score, precision_score, confusion_matrix

from transformer.modeling import BertForSequenceClassification
from transformer.tokenization import BertTokenizer
from transformer.optimization import BertAdam
from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME

#csv.field_size_limit(sys.maxsize)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('debug_layer_loss.log')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()

class EarlyStopper:
    def __init__(self, patience=15, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

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

class MLMAProcessor(DataProcessor):
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
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev")

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
        if ex_index % 1000 == 0:
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

        label_id = label_map[example.label]
        
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


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_metrics(task_name, preds, labels):
    results = {}
    if task_name.lower() == 'mlma':
        results['acc'] = simple_accuracy(preds, labels)
        results['recall'] = recall_score(labels, preds)
        results['precision'] = precision_score(labels, preds)
        results['f1_score'] = f1_score(labels, preds)
        results['f1_weighted'] = f1_score(labels, preds, average = 'weighted')
        results['confusion_matrix'] = confusion_matrix(labels, preds)
    return results
    

def do_eval(model, task_name, eval_dataloader,
            device, eval_labels, num_labels):
    eval_loss = 0
    nb_eval_steps = 0
    preds = []

    for batch_ in tqdm(eval_dataloader, desc="Evaluating"):
        batch_ = tuple(t.to(device) for t in batch_)
        with torch.no_grad():
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_

            logits = model(input_ids, segment_ids, input_mask)

        # create eval loss and other metric required by the task
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
    preds = np.argmax(preds, axis=1)

    result = compute_metrics(task_name, preds, eval_labels.numpy())
    result['eval_loss'] = eval_loss

    return result

def get_tensor_data(features):
    
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_label_ids, all_seq_lengths)
    return tensor_data, all_label_ids

def result_to_file(result, file_name):
    with open(file_name, "a") as writer:
        logger.info("***** Eval results *****")
        if type(result) == str:
            writer.write("%s\n" % (result))
            return
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--pre_trained_bert",
                        default=None,
                        type=str,
                        required=True,
                        help="The pre-trained bert model dir.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    
    
    parser.add_argument("--eval_mode",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay', '--wd',
                        default=1e-2,
                        type=float,
                        metavar='W',
                        help='weight decay')
    parser.add_argument("--num_train_epochs",
                        default=1.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument('--eval_step',
                        type=int,
                        default=10)

    args = parser.parse_args()
    logger.info('The args: {}'.format(args))

    processors = {
        "MLMA": MLMAProcessor
    }

    # info regarding the availability of GPUs
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    if args.eval_mode:
        ### be sure to give the updated model files directory for this block to work properly
        processor = processors[args.task_name]()
        label_list = processor.get_labels()
        num_labels = len(label_list)
        tokenizer = BertTokenizer.from_pretrained(args.pre_trained_bert, do_lower_case=args.do_lower_case)

        eval_examples = processor.get_test_examples(args.data_dir)
        eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer)
        eval_data, eval_labels = get_tensor_data(eval_features)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model = BertForSequenceClassification.from_pretrained(args.pre_trained_bert, num_labels=num_labels)
        model.to(device)

        model.eval()
        result = do_eval(model, args.task_name, eval_dataloader,
                                 device, eval_labels, num_labels)
        
        output_eval_file = os.path.join(args.output_dir, "test_results.txt")
        result_to_file("###### test results are \n", output_eval_file)
        result_to_file(result, output_eval_file)
        return

        


    # Prepare seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    
    # Prepare task settings
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)



    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.pre_trained_bert, do_lower_case=args.do_lower_case)

    # getting data ready
    train_examples = processor.get_train_examples(args.data_dir)
    if args.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps))
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    
    ### Training Set
    train_features = convert_examples_to_features(train_examples, label_list,
                                                      args.max_seq_length, tokenizer)
    train_data, _ = get_tensor_data(train_features)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    ### Dev Set
    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer)
    eval_data, eval_labels = get_tensor_data(eval_features)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    ### Loading the model
    model = BertForSequenceClassification.from_pretrained(args.pre_trained_bert, num_labels=num_labels)
    model.to(device)

    ### Prepare the optimizer
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    size = 0
    for n, p in model.named_parameters():
        logger.info('n: {}'.format(n))
        size += p.nelement()

    logger.info('Total parameters: {}'.format(size))
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    schedule = 'warmup_linear'
    
    optimizer = BertAdam(optimizer_grouped_parameters,
                            schedule=schedule,
                            lr=args.learning_rate,
                            warmup=args.warmup_proportion,
                            t_total=num_train_optimization_steps)
    
    
    # Train and evaluate
    global_step = 0
    best_dev_acc = 0.0
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    loss_func = CrossEntropyLoss(weight = torch.tensor([8.3, 1.]).to(device))
    early_stopper = EarlyStopper()
    #loss_func = CrossEntropyLoss()
    for epoch_ in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0.
        model.train()
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch
            if input_ids.size()[0] != args.train_batch_size:
                continue
            cls_loss = 0.
            logits = model(input_ids, segment_ids, input_mask)
            #print(logits.is_cuda)
            #print(label_ids.is_cuda)
            loss = loss_func(logits.view(-1, num_labels), label_ids.view(-1))

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += label_ids.size(0)
            nb_tr_steps += 1

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            # if (global_step + 1) % args.eval_step == 0:
            #     logger.info("***** Running evaluation *****")
            #     logger.info("  Process = {} iter {} step".format(epoch_, global_step))
            #     logger.info("  Num examples = %d", len(eval_examples))
            #     logger.info("  Batch size = %d", args.eval_batch_size)

            #     model.eval()
            #     loss = tr_loss / (step + 1)

            #     result = do_eval(model, args.task_name, eval_dataloader,
            #                      device, eval_labels, num_labels)
                
            #     result['global_step'] = global_step
            #     result['loss'] = loss
            #     result_to_file(result, output_eval_file)

            #     save_model = False
            #     if result['acc'] > best_dev_acc:
            #         best_dev_acc = result['acc']
            #         save_model = True

            #     if save_model:
            #         logger.info("***** Save model *****")

            #         model_to_save = model.module if hasattr(model, 'module') else model

            #         model_name = WEIGHTS_NAME
            #         # if not args.pred_distill:
            #         #     model_name = "step_{}_{}".format(global_step, WEIGHTS_NAME)
            #         output_model_file = os.path.join(args.output_dir, model_name)
            #         output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

            #         torch.save(model_to_save.state_dict(), output_model_file)
            #         model_to_save.config.to_json_file(output_config_file)
            #         tokenizer.save_vocabulary(args.output_dir)

            #     model.train()

        ##### evluating after epoch
        model.eval()
        result = do_eval(model, args.task_name, eval_dataloader,
                                 device, eval_labels, num_labels)
        state = early_stopper.early_stop(result['eval_loss'])
        result_to_file("###### epoch {} results are \n".format(epoch_), output_eval_file)
        result_to_file(result, output_eval_file)
        if state:
            break
        if early_stopper.counter == 0:
            logger.info("***** Saving best model till now *****")
            result_to_file("###### saving the model at epoch {} \n".format(epoch_), output_eval_file)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_name = WEIGHTS_NAME
            output_model_file = os.path.join(args.output_dir, model_name)
            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(args.output_dir)
            
        model.train()




if __name__ == "__main__":
    main()













    



    
