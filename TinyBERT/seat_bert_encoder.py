''' Convenience functions for handling BERT '''
import torch
from transformer.modeling import BertModel, BertForSequenceClassification, TinyBertForSequenceClassification
from transformer.tokenization import BertTokenizer

def load_model(version='bert-base-uncased'):
    ''' Load BERT model and corresponding tokenizer '''
    tokenizer = BertTokenizer.from_pretrained(version, do_lower_case= True)
    model = TinyBertForSequenceClassification.from_pretrained(version, num_labels=2)
    model.eval()

    return model, tokenizer


def encode(model, tokenizer, texts):
    ''' Use tokenizer and model to encode texts '''
    encs = {}
    for text in texts:
        #print(text)
        tokenized = tokenizer.tokenize(text)
        indexed = tokenizer.convert_tokens_to_ids(tokenized)
        segment_idxs = [0] * len(tokenized)
        tokens_tensor = torch.tensor([indexed])
        segments_tensor = torch.tensor([segment_idxs])
        _, _, enc = model(tokens_tensor, segments_tensor)
        enc = enc[-1]
        #print(enc.shape)
        enc = enc[:, 0, :]  # extract the last rep of the first input
        encs[text] = enc.detach().view(-1).numpy()
    return encs
