import json
import random
import numpy as np
from torch.utils.data import Dataset
import torch
from tqdm import tqdm

# We follow the BILUO scheme
# B-egin: The first token of a multi-token entity
# I-n: An inner token of a multi-token entity
# L-ast: The final token of a multi-token entity
# U-nit: A single-token entity
# O-ut: A non-entity token
label2id = {"B": 0,
            "I": 1,
            "L": 2,
            "U": 3,
            "O": 4
            }

id2label = {0: "B",
            1: "I",
            2: "L",
            3: "U",
            4: "O"
            }


def map_labels_to_text(text, segments):
    """
    Takes as input a tokenized string, as well as a list of tokenized segments that contain values to be annotated
    :param text: text to be labeled
    :param segments: list of text segments that require value label
    :returns: labels for each token of the input text
    """
    labels = np.full(text.shape, label2id["O"])
    for s in segments:

        for i in range(1, len(text)-len(s)):
            text_segment = text[i:i+len(s)]
            if np.array_equal(text_segment, s):
                for j in range(len(s)):
                    if j == 0:
                        labels[i] = label2id["B"]
                    else:
                        labels[i+j] = label2id["I"]

    return labels


def load_dataset(dataset_path, tokenizer, for_testing_purposes=False):
    data = []
    dialogs = json.load(open(dataset_path))
    for i, dialog in enumerate(dialogs):
        if for_testing_purposes and i > 10:
            break
        for utterance in dialog['utterances']:
            tokens = tokenizer(utterance['text'], return_tensors="np", truncation=True)
            tokenized_text = tokens['input_ids'][0]
            attention_mask = tokens['attention_mask'][0]
            token_type_ids = tokens['token_type_ids'][0]
            segments = []
            if 'segments' in utterance:
                for seg in utterance['segments']:
                    segments.append(tokenizer(seg['text'], return_tensors="np")['input_ids'][0][1:-1])
            label = map_labels_to_text(tokenized_text, segments)
            data.append({"input_ids": tokenized_text,
                         "attention_mask": attention_mask,
                         "token_type_ids": token_type_ids,
                         "labels": label,
                         "text": utterance['text']
                         })
    return data


def load_taskmaster_datasets(datasets, tokenizer, for_testing_purposes=True, train_percent=1, shuffle=True):
    """
    Load all data from Taskmaster 1
    :returns: data - a list of tokenized text, label pairs
    """
    data = []
    for dataset in tqdm(datasets):
        data.extend(load_dataset(dataset, tokenizer, for_testing_purposes))

    if shuffle:
        random.shuffle(data)
    train_data = data[:round(len(data)*train_percent)]
    val_data = data[round(len(data)*train_percent):]

    return train_data, val_data


class taskmaster_dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {"input_ids": self.data[index]['input_ids'],
                "attention_mask": self.data[index]['attention_mask'],
                "token_type_ids": self.data[index]['token_type_ids'],
                "labels": self.data[index]['labels'],
                "text": self.data[index]['text']}


def pad_sequence(sequence, max_seq_len, pad_value):
    return np.pad(sequence, (0, max_seq_len-len(sequence)), 'constant', constant_values=pad_value)


class collate_class(object):
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max([len(sample['labels']) for sample in batch])
        batch_input_ids = []
        batch_attention_mask = []
        batch_token_type_ids = []
        batch_labels = []
        batch_text = []

        for item in batch:
            batch_input_ids.append(pad_sequence(item['input_ids'], max_len, self.pad_token_id))
            batch_attention_mask.append(pad_sequence(item['attention_mask'], max_len, 0))
            batch_token_type_ids.append(pad_sequence(item['token_type_ids'], max_len, 0))
            batch_labels.append(pad_sequence(item['labels'], max_len, label2id["O"]))
            batch_text.append(item['text'])

        return {
            "input_ids": torch.tensor(batch_input_ids),
            "attention_mask": torch.tensor(batch_attention_mask),
            "token_type_ids": torch.tensor(batch_token_type_ids),
            "labels": torch.tensor(batch_labels),
            "text": batch_text
        }
