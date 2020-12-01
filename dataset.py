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

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]


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


def load_multiwoz_dataset(dataset_path, tokenizer, slots, for_testing_purposes=False):
    data = []
    dialogs = json.load(open(dataset_path))

    samples_per_slot = {slot: 0 for slot in slots}

    for i, dialog_dict in enumerate(dialogs):
        if for_testing_purposes and i > 10:
            break
        skip = False
        for domain in dialog_dict['domains']:
            if domain not in EXPERIMENT_DOMAINS:
                skip = True

        if not skip:
            for turn in dialog_dict['dialogue']:
                # tokens = tokenizer(turn['system_transcript'], return_tensors="np", truncation=True)
                # tokenized_text = tokens['input_ids'][0]
                # attention_mask = tokens['attention_mask'][0]
                # token_type_ids = tokens['token_type_ids'][0]
                # turn_labels = [tokenizer(v, return_tensors="np")['input_ids'][0][1:-1] for ds, v in turn['turn_label']]
                # label = map_labels_to_text(tokenized_text, turn_labels)
                # data.append({"input_ids": tokenized_text,
                #              "attention_mask": attention_mask,
                #              "token_type_ids": token_type_ids,
                #              "labels": label,
                #              "text": turn['system_transcript']
                #              })

                tokens = tokenizer(turn['transcript'], return_tensors="np", truncation=True)
                tokenized_text = tokens['input_ids'][0]
                attention_mask = tokens['attention_mask'][0]
                token_type_ids = tokens['token_type_ids'][0]
                for ds, v in turn['turn_label']:
                    if ds in slots:
                        samples_per_slot[ds] += 1
                turn_labels = [tokenizer(v, return_tensors="np")['input_ids'][0][1:-1] for ds, v in turn['turn_label'] if ds in slots]
                label = map_labels_to_text(tokenized_text, turn_labels)
                data.append({"input_ids": tokenized_text,
                             "attention_mask": attention_mask,
                             "token_type_ids": token_type_ids,
                             "labels": label,
                             "text": turn['transcript']
                             })

    # for d in data:
    #     for t, l in zip(d['input_ids'], d['labels']):
    #         print(tokenizer.decode([t]), l)
    print("Count of samples for {}".format(dataset_path))
    for ds, count in samples_per_slot.items():
        print(f"{ds} - {count}")
    return data


def load_MW_22_dataset_current_turn_beliefs(dataset_path, tokenizer, slots, for_testing_purposes=False):
    """
    This will load the MultiWOZ 2.2 dataset
    It finds those slot-values which are new to the belief state during each turn
            and considers only those which are new
    """
    data = []
    samples_per_slot = {slot: 0 for slot in slots}
    dialogs = json.load(open(dataset_path))
    for i, dialog in enumerate(dialogs):
        if for_testing_purposes and i > 10:
            break
        skip = False
        for domain in dialog['services']:
            if domain not in EXPERIMENT_DOMAINS:
                skip = True
        if not skip:
            # track turn values
            system_turn_values = {}

            # track the overall dialogue state so that we know what exists in the current turn
            current_turn_beliefs = {}
            prev_turn_beliefs = {}

            # track the dialogue for invididual turns (each turn in SYS utterance, then USR utterance)
            turn_dialog = "[SYS] "

            for turn in dialog['turns']:
                if turn['speaker'] == "SYSTEM":
                    turn_dialog = f"[SYS] {turn['utterance']} "
                    system_turn_values = get_slot_values_from_SYS_MW_22(turn, slots)

                if turn['speaker'] == "USER":
                    turn_dialog += f"[USR] {turn['utterance']} "
                    user_turn_values = get_slot_values_from_USR_MW_22(turn, slots)

                    belief_state = system_turn_values
                    belief_state.update(user_turn_values)

                    # calculate beliefs that have just been updated this turn
                    current_turn_beliefs = {}
                    for k in belief_state:
                        if k not in prev_turn_beliefs:
                            current_turn_beliefs[k] = belief_state[k]
                        if k in prev_turn_beliefs:
                            for v in belief_state[k]:
                                if v not in prev_turn_beliefs[k]:
                                    if k not in current_turn_beliefs:
                                        current_turn_beliefs[k] = []
                                    current_turn_beliefs[k].append(v)

                    prev_turn_beliefs = belief_state

                    turn_values = set()
                    for ds, vals in current_turn_beliefs.items():
                        if ds in slots:
                            for val in vals:
                                # some values do not exist in the dialogue
                                # eg. if there are multiple possible values (03:00, 3:00)
                                #   or if the value comes from a previous turn
                                if val not in turn_dialog.lower():
                                    a = 1
                                else:
                                    samples_per_slot[ds] += 1
                                    turn_values.add(val)

                    tokens = tokenizer(turn_dialog, return_tensors="np", truncation=True)
                    tokenized_text = tokens['input_ids'][0]
                    attention_mask = tokens['attention_mask'][0]
                    token_type_ids = tokens['token_type_ids'][0]
                    text = tokenizer.decode(tokenized_text)

                    turn_labels = [tokenizer(v, return_tensors="np")['input_ids'][0][1:-1] for v in turn_values]
                    for label in turn_labels:
                        t = tokenizer.decode(label)
                    label = map_labels_to_text(tokenized_text, turn_labels)
                    data.append({"input_ids": tokenized_text,
                                 "attention_mask": attention_mask,
                                 "token_type_ids": token_type_ids,
                                 "labels": label,
                                 "text": turn_dialog
                                 })

    # for d in data:
    #     print("==================")
    #     for t, l in zip(d['input_ids'], d['labels']):
    #         tab = 20-l.item()
    #         print(f"{tokenizer.decode([t]):<{tab}} {l}")
    print("Count of samples for {}".format(dataset_path))
    for ds, count in samples_per_slot.items():
        print(f"{ds} - {count}")
    return data


def load_MW_22_dataset_full_belief_state(dataset_path, tokenizer, slots, for_testing_purposes=False):
    """
    This will load the MultiWOZ 2.2 dataset
    It considers the entire belief state at each turn, and labels any slot-values which 
            are found in the turn dialogue
    """
    
    data = []
    samples_per_slot = {slot: 0 for slot in slots}
    dialogs = json.load(open(dataset_path))
    for i, dialog in enumerate(dialogs):
        if for_testing_purposes and i > 10:
            break
        skip = False
        for domain in dialog['services']:
            if domain not in EXPERIMENT_DOMAINS:
                skip = True
        if not skip:
            # track turn values
            system_turn_values = {}

            # track the dialogue for invididual turns (each turn in SYS utterance, then USR utterance)
            turn_dialog = "[SYS] "

            for turn in dialog['turns']:
                if turn['speaker'] == "SYSTEM":
                    turn_dialog = f"[SYS] {turn['utterance']} "
                    system_turn_values = get_slot_values_from_SYS_MW_22(turn, slots)

                if turn['speaker'] == "USER":
                    turn_dialog += f"[USR] {turn['utterance']} "
                    user_turn_values = get_slot_values_from_USR_MW_22(turn, slots)

                    belief_state = system_turn_values
                    belief_state.update(user_turn_values)

                    tokens = tokenizer(turn_dialog, return_tensors="np", truncation=True)
                    tokenized_text = tokens['input_ids'][0]
                    attention_mask = tokens['attention_mask'][0]
                    token_type_ids = tokens['token_type_ids'][0]
                    text = tokenizer.decode(tokenized_text)

                    turn_labels = []
                    for ds, vals in belief_state.items():
                        if ds in slots:
                            for v in vals:
                                if v in turn_dialog:
                                    samples_per_slot[ds] += 1
                                turn_labels.append(tokenizer(v, return_tensors="np")['input_ids'][0][1:-1])

                    label = map_labels_to_text(tokenized_text, turn_labels)

                    # for debugging
                    # print("=======================")
                    # print(belief_state)
                    # print("=======================")
                    # for t, label in zip(tokenized_text, label):
                    #     print(f"{tokenizer.decode([t]):<{20-label.item()}} {label}")

                    data.append({"input_ids": tokenized_text,
                                 "attention_mask": attention_mask,
                                 "token_type_ids": token_type_ids,
                                 "labels": label,
                                 "text": turn_dialog
                                 })

    print("Count of samples for {}".format(dataset_path))
    for ds, count in samples_per_slot.items():
        print(f"{ds} - {count}")
    return data


def load_MW_22_dataset_training(tokenizer, slots, for_testing_purposes=False):
    files = ["MultiWOZ_2.2/train/dialogues_001.json", "MultiWOZ_2.2/train/dialogues_002.json",
             "MultiWOZ_2.2/train/dialogues_003.json", "MultiWOZ_2.2/train/dialogues_004.json",
             "MultiWOZ_2.2/train/dialogues_005.json", "MultiWOZ_2.2/train/dialogues_006.json",
             "MultiWOZ_2.2/train/dialogues_007.json", "MultiWOZ_2.2/train/dialogues_008.json",
             "MultiWOZ_2.2/train/dialogues_009.json", "MultiWOZ_2.2/train/dialogues_010.json",
             "MultiWOZ_2.2/train/dialogues_011.json", "MultiWOZ_2.2/train/dialogues_012.json",
             "MultiWOZ_2.2/train/dialogues_013.json", "MultiWOZ_2.2/train/dialogues_014.json",
             "MultiWOZ_2.2/train/dialogues_015.json", "MultiWOZ_2.2/train/dialogues_016.json",
             "MultiWOZ_2.2/train/dialogues_017.json", ]

    data = []
    for file in files:
        data.extend(load_MW_22_dataset_full_belief_state(file, tokenizer, slots, for_testing_purposes))

    return data


def load_MW_22_dataset_validation(tokenizer, slots, for_testing_purposes=False):
    files = ["MultiWOZ_2.2/dev/dialogues_001.json", "MultiWOZ_2.2/dev/dialogues_002.json"]

    data = []
    for file in files:
        data.extend(load_MW_22_dataset_full_belief_state(file, tokenizer, slots, for_testing_purposes))

    return data


def load_taskmaster_dataset(dataset_path, tokenizer, for_testing_purposes=False):
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
        data.extend(load_taskmaster_dataset(dataset, tokenizer, for_testing_purposes))

    if shuffle:
        random.shuffle(data)
    train_data = data[:round(len(data)*train_percent)]
    val_data = data[round(len(data)*train_percent):]

    return train_data, val_data


class VE_dataset(Dataset):
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


def get_slot_values_from_SYS_MW_22(turn, slots):
    """Given the systems turn, return any upates to the belief state
    For normalization purposes, even though slot values are strings in this situation,
            this function will return them as lists

    The only belief states which can be updated in the system turn
        are non-categorical slots

    Each slot in a frame follows exactly the schema

        {
        "slots": [
            {
            "slot": String of slot name.
            "start": Int denoting the index of the starting character in the utterance corresponding to the slot value.
            "exclusive_end": Int denoting the index of the character just after the last character corresponding to the slot value in the utterance. In python, utterance[start:exclusive_end] gives the slot value.
            "value": String of value. It equals to utterance[start:exclusive_end], where utterance is the current utterance in string.
            }
        ]
        }

    """

    system_values = {}

    for frame in turn['frames']:

        # Speaker turns NEVER have an action
        if frame['actions']:
            a = 1

        # speaker service is ALWAYS in experiment domains
        if frame['service'] not in EXPERIMENT_DOMAINS:
            a = 1

        for slot in frame['slots']:
            if slot['slot'] in slots:
                system_values[slot['slot']] = [slot['value'].lower()]

    return system_values


def get_slot_values_from_USR_MW_22(turn, slots):
    """Given the users turn, return the belief state

    within a given turn, the data is split into "frames" aka domains, and each frame has
            the following data

    usr_slots are some of the non-categorical slots that are filled during this turn
            some of the values in usr_slots are copied from other slots

    slots from usr_slots will follow one of the following schemas:

        {
        "slots": [
            {
            "slot": String of slot name.
            "start": Int denoting the index of the starting character in the utterance corresponding to the slot value.
            "exclusive_end": Int denoting the index of the character just after the last character corresponding to the slot value in the utterance. In python, utterance[start:exclusive_end] gives the slot value.
            "value": String of value. It equals to utterance[start:exclusive_end], where utterance is the current utterance in string.
            }
        ]
        }

    OR

        {
        "slots": [
            {
            "slot": Slot name string.
            "copy_from": The slot to copy from.
            "value": A list of slot values being . It corresponds to the state values of the "copy_from" slot.
            }
        ]
        }

    belief_state_frame usually contains all of the belief state for a given turn for a particular domain(aka frame)
            however, sometimes values exist in the utterance, and the usr_slot and not the belief_state

    belief_state_frame follows the following schema:

        {
        "state":{
            "active_intent": String. User intent of the current turn.
            "requested_slots": List of string representing the slots, the values of which are being requested by the user.
            "slot_values": Dict of state values. The key is slot name in string. The value is a list of values.
        }
        }

    where each value in slot values is a list


    """

    belief_state = {}
    for frame in turn['frames']:

        # disgregard domains outside of the experiment domain
        if frame['service'] not in EXPERIMENT_DOMAINS:
            continue

        usr_slots = {}
        # check for non-categorical slots
        # these values could be copied from another slot, if 'copy_from' in slot.keys()
        # slot values here will be exactly as they are found in the text (uppercase, etc.)
        for slot in frame['slots']:
            if type(slot['value']) == str:
                a = 1
            usr_slots[slot['slot']] = slot['value']

        # get belief state
        # each v is a list of matching values for a single slot (eg. [9:00, 09:00])
        #   sometimes only a single of the values appears in the dialogue, sometimes both values appear
        belief_state_frame = frame['state']['slot_values']

        # Sometimes, a slot value for the current turn comes from usr_slots
        # and won't be added to the belief state until the next turn
        # it happens 383 times, compared with 33047 times where the slot value is already in the belief state
        # Here, I decided to just manually add them to the belief state
        for k, v in usr_slots.items():
            if k not in belief_state_frame.keys():
                if type(v) == str:
                    a = 1
                belief_state_frame[k] = [v.lower()]

        belief_state.update(belief_state_frame)

    # To track slots with multiple true values
    # for k, v in belief_state.items():
    #     if len(v) > 1:
    #         print(v)

    filtered_belief_state = {}
    for slot in belief_state:
        if slot in slots:
            filtered_belief_state[slot] = belief_state[slot]

    return filtered_belief_state
