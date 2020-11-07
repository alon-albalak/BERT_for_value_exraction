from transformers import BertTokenizer, BertForTokenClassification, BertConfig
from torch.utils.data import DataLoader
import torch
from dataset import load_TM_1_data, TM_1_dataset, collate_class

label2id = {"B":0,
            "I":1,
            "L":2,
            "U":3,
            "O":4
            }

id2label = {0:"B",
            1:"I",
            2:"L",
            3:"U",
            4:"O"
            }

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(label2id.keys()), return_dict=True)
model.to('cuda')

data = load_TM_1_data(tokenizer, temp_var=True)
dataset = TM_1_dataset(data)
collator = collate_class(tokenizer.pad_token_id, device='cuda')
dataloader = DataLoader(dataset, batch_size = 4, shuffle=False, collate_fn=collator)

for i, batch in enumerate(dataloader):
    if i < 4:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        labels = batch['labels']
        # print(f"input_ids: {input_ids.shape} - {input_ids}")
        # print(f"attention_mask: {attention_mask.shape} - {attention_mask}")
        # print(f"token_type_ids: {token_type_ids.shape} - {token_type_ids}")
        # print(f"labels: {labels.shape} - {labels}")
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        # print(f"loss: {loss}")
        # print(f"logits: {logits}")

        

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# print(inputs)
# labels = torch.tensor([1] * inputs["input_ids"].size(1)).unsqueeze(0)  # Batch size 1

# outputs = model(**inputs, labels=labels)
# loss = outputs.loss
# logits = outputs.logits

# print(tokenizer.decode(inputs['input_ids'].squeeze()))