import os
import numpy as np
import torch
from pytorch_transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from tqdm import tqdm


class BertKemenkunhamSentimentClassification():
    def __init__(self,
                 labels_list,
                 models_dir,
                 gpu_id):
        self.labels_list = labels_list
        if gpu_id == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(f'cuda:{gpu_id}')

        self.bert_config = BertConfig.from_pretrained(models_dir)
        if len(labels_list) != self.bert_config.num_labels:
            raise Exception(
                f'Different length of labels_list versus pre-trained labels_num on Bert Model !! ({len(labels_list)} vs {self.bert_config.num_labels})')
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            models_dir, do_lower_case=False)
        self.bert_model = BertForSequenceClassification.from_pretrained(
            models_dir)
        self.bert_model.to(self.device)

        self.bert_model.eval()

    def predict(self, text):
        features = self.convert_input_to_features(text,
                                                  512,
                                                  self.bert_tokenizer,
                                                  cls_token_at_end=False,
                                                  cls_token=self.bert_tokenizer.cls_token,
                                                  sep_token=self.bert_tokenizer.sep_token,
                                                  cls_token_segment_id=0,
                                                  pad_on_left=False,
                                                  pad_token_segment_id=0)

        with torch.no_grad():

            input_ids = torch.tensor(
                [f.input_ids for f in features], dtype=torch.long).to(self.device)

            input_mask = torch.tensor(
                [f.input_mask for f in features], dtype=torch.long).to(self.device)
            segment_ids = torch.tensor(
                [f.segment_ids for f in features], dtype=torch.long).to(self.device)
            inputs = {'input_ids':      input_ids,
                      'attention_mask': input_mask,
                      # XLM don't use segment_ids
                      'token_type_ids': segment_ids,
                      'labels':         None}

            outputs = self.bert_model(**inputs)
            log_softmax = torch.nn.Softmax(dim=1)
            scores = log_softmax(outputs[0]).cpu().numpy()

            result = {}
            for idx, score in enumerate(scores[0]):
                result[self.labels_list[idx]] = score
        return result

    def convert_input_to_features(self,
                                  text,
                                  max_seq_length,
                                  tokenizer,
                                  cls_token_at_end=False,
                                  cls_token='[CLS]',
                                  cls_token_segment_id=1,
                                  sep_token='[SEP]',
                                  sep_token_extra=False,
                                  pad_on_left=False,
                                  pad_token=0,
                                  pad_token_segment_id=0,
                                  sequence_a_segment_id=0,
                                  sequence_b_segment_id=1,
                                  mask_padding_with_zero=True):
        tokens = tokenizer.tokenize(text)
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
        tokens = tokens + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1]
                          * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] *
                           padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + \
                ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + \
                ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        features = []
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=None))
        return features


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


if __name__ == "__main__":
    sentiment_classifier = BertKemenkunhamSentimentClassification(
        gpu_id=0,
        models_dir='models',
        labels_list=['netral', 'negatif', 'positif']
    )

    with open('dataset/sentiment_val.txt', 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    import re
    import operator
    from sklearn.metrics import accuracy_score
    from tqdm import tqdm

    def preprocess(text):
        text = ''.join(text.split('WIB')[1:])
        pattern = r'\[.*?\]'
        text = re.sub(pattern, '', text)
        pattern = r'\(.*?\)'
        text = re.sub(pattern, '', text)
        pattern = r'\<.*?\>'
        text = re.sub(pattern, '', text)

        return text

    label_list = ['netral', 'negatif', 'positif']
    gts = []
    preds = []
    for line in tqdm(lines):
        gt = label_list.index(line.split(' ')[-1])

        with open(os.path.join('dataset/news_data', line.split(' ')[0]), 'r') as f:
            text = preprocess(f.read())

        result = sentiment_classifier.predict(text)

        pred = label_list.index(
            max(result.items(), key=operator.itemgetter(1))[0])
        gts.append(gt)
        preds.append(pred)

    print(
        f'Accuracy from validation data : {accuracy_score(gts, preds)*100} %')
