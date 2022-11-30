''' Main script for loading models and running WEAT tests '''

import os
import sys
import random
import re
import argparse
import logging as log
import json
log.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d %I:%M:%S %p', level=log.INFO)  # noqa

from csv import DictWriter
from enum import Enum

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import weat_score as weat
import seat_bert_encoder as bert

TEST_EXT = '.jsonl'
BERT_VERSIONS = ["bert-base-uncased", "bert-large-uncased", "bert-base-cased", "bert-large-cased", "tinybert_model"]


def load_json(sent_file):
    ''' Load from json. We expect a certain format later, so do some post processing '''
    log.info("Loading %s..." % sent_file)
    all_data = json.load(open(sent_file, 'r'))
    data = {}
    for k, v in all_data.items():
        examples = v["examples"]
        data[k] = examples
        v["examples"] = examples
    return all_data  # data

def test_sort_key(test):
    '''
    Return tuple to be used as a sort key for the specified test name.
   Break test name into pieces consisting of the integers in the name
    and the strings in between them.
    '''
    key = ()
    prev_end = 0
    for match in re.finditer(r'\d+', test):
        key = key + (test[prev_end:match.start()], int(match.group(0)))
        prev_end = match.end()
    key = key + (test[prev_end:],)

    return key


def split_comma_and_check(arg_str, allowed_set, item_type):
    ''' Given a comma-separated string of items,
    split on commas and check if all items are in allowed_set.
    item_type is just for the assert message. '''
    items = arg_str.split(',')
    for item in items:
        if item not in allowed_set:
            raise ValueError("Unknown %s: %s!" % (item_type, item))
    return items

def handle_arguments(arguments):
    ''' Helper function for handling argument parsing '''
    parser = argparse.ArgumentParser(
        description='Run specified SEAT tests on specified models.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tests', '-t', type=str,
                        help="WEAT tests to run (a comma-separated list; test files should be in `data_dir` and "
                             "have corresponding names, with extension {}). Default: all tests.".format(TEST_EXT))
    parser.add_argument('--model', '-m', type=str,
                        help="Model to evaluate",
                        default='bert')
    parser.add_argument('--seed', '-s', type=int, help="Random seed", default=1111)
    parser.add_argument('--log_file', '-l', type=str,
                        help="File to log to")
    parser.add_argument('--ignore_cached_encs', '-i', action='store_true',
                        help="If set, ignore existing encodings and encode from scratch.")
    parser.add_argument('--dont_cache_encs', action='store_true',
                        help="If set, don't cache encodings to disk.")
    parser.add_argument('--data_dir', '-d', type=str,
                        help="Directory containing examples for each test",
                        default='../SEAT/tests')
    parser.add_argument('--n_samples', type=int,
                        help="Number of permutation test samples used when estimate p-values (exact test is used if "
                             "there are fewer than this many permutations)",
                        default=100000)
    parser.add_argument('--parametric', action='store_true',
                        help='Use parametric test (normal assumption) to compute p-values.')
    parser.add_argument('--results_path', type=str,
                        help="Path where TSV results file will be written")
    bert_group = parser.add_argument_group('bert', 'Options for BERT model')
    bert_group.add_argument('--bert_version', type=str,
                            help="Version of BERT to use or path to model.", default="bert-base-uncased")                     
    return parser.parse_args(arguments)

def maybe_make_dir(dirname):
    ''' Maybe make directory '''
    os.makedirs(dirname, exist_ok=True)

def main(arguments):
    ''' Main logic: parse args for tests to run and which models to evaluate '''
    log.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d %I:%M:%S %p', level=log.INFO)

    args = handle_arguments(arguments)
    if args.seed >= 0:
        log.info('Seeding random number generators with {}'.format(args.seed))
        random.seed(args.seed)
        np.random.seed(args.seed)
    if args.log_file:
        log.getLogger().addHandler(log.FileHandler(args.log_file))
    log.info("Parsed args: \n%s", args)

    all_tests = sorted(
        [
            entry[:-len(TEST_EXT)]
            for entry in os.listdir(args.data_dir)
            if not entry.startswith('.') and entry.endswith(TEST_EXT)
        ],
        key=test_sort_key
    )
    log.debug('Tests found:')
    for test in all_tests:
        log.debug('\t{}'.format(test))

    tests = split_comma_and_check(args.tests, all_tests, "test") if args.tests is not None else all_tests
    log.info('Tests selected:')
    for test in tests:
        log.info('\t{}'.format(test))
    model_name = args.model
    log.info('\t{}'.format(model_name))

    results = []
    log.info('Running tests for model {}'.format(model_name))
    if model_name == 'bert':
        model_options = 'version=' + args.bert_version
    else:
        raise ValueError("Model %s not found!" % model_name)
    model = None
    
    for test in tests:
        # load the test data
        encs = load_json(os.path.join(args.data_dir, "%s%s" % (test, TEST_EXT)))
        log.info('Running test {} for model {}'.format(test, model_name))
        if model_name == 'bert':
            # load the model and do model-specific encoding procedure
            log.info('Computing sentence encodings')
            model, tokenizer = bert.load_model(args.bert_version)
            encs_targ1 = bert.encode(model, tokenizer, encs["targ1"]["examples"])
            encs_targ2 = bert.encode(model, tokenizer, encs["targ2"]["examples"])
            encs_attr1 = bert.encode(model, tokenizer, encs["attr1"]["examples"])
            encs_attr2 = bert.encode(model, tokenizer, encs["attr2"]["examples"])
            encs["targ1"]["encs"] = encs_targ1
            encs["targ2"]["encs"] = encs_targ2
            encs["attr1"]["encs"] = encs_attr1
            encs["attr2"]["encs"] = encs_attr2
        else:
            raise ValueError("Model %s not found!" % model_name)

        log.info("\tDone!")

        enc = [e for e in encs["targ1"]['encs'].values()][0]
        d_rep = enc.size if isinstance(enc, np.ndarray) else len(enc)

        # run the test on the encodings
        log.info("Running SEAT...")
        log.info("Representation dimension: {}".format(d_rep))
        esize, pval = weat.run_test(encs, n_samples=args.n_samples, parametric=args.parametric)
        results.append(dict(
            model=model_name,
            options=model_options,
            test=test,
            p_value=pval,
            effect_size=esize,
            num_targ1=len(encs['targ1']['encs']),
            num_targ2=len(encs['targ2']['encs']),
            num_attr1=len(encs['attr1']['encs']),
            num_attr2=len(encs['attr2']['encs'])))

        log.info("Model: %s", model_name)
        log.info('Options: {}'.format(model_options))
        for r in results:
            log.info("\tTest {test}:\tp-val: {p_value:.9f}\tesize: {effect_size:.2f}".format(**r))

    if args.results_path is not None:
        log.info('Writing results to {}'.format(args.results_path))
        with open(args.results_path, 'w') as f:
            writer = DictWriter(f, fieldnames=results[0].keys(), delimiter='\t')
            writer.writeheader()
            for r in results:
                writer.writerow(r)


if __name__ == "__main__":
    main(sys.argv[1:])