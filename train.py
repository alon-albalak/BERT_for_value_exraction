from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertForTokenClassification
from dataset import load_TM_1_data, TM_1_dataset, collate_class
from BertForValueExtraction import BertForValueExtraction

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

batch_size = 16
gradient_accumulation_steps = 32/batch_size
initial_lr = 1e-5

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_data, val_data = load_TM_1_data(tokenizer, train_percent=0.9)

train_dataset = TM_1_dataset(train_data)
val_dataset = TM_1_dataset(val_data)

collator = collate_class(tokenizer.pad_token_id, device='cuda')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)

# Regular version
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(label2id.keys()), return_dict=True)
model.train()
model.to('cuda')


optimizer = torch.optim.Adam(params=model.parameters(), lr=initial_lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, min_lr=initial_lr/100, verbose=True)

for epoch in range(1):
    # train loop
    total_loss = 0
    optimizer.zero_grad()
    pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for i, batch in pbar:

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
        loss.backward()
        total_loss += loss.item()
        if ((i+1) % gradient_accumulation_steps) == 0:
            optimizer.step()
            optimizer.zero_grad()

        batch_num = ((i+1)/gradient_accumulation_steps)
        pbar.set_description(f"Loss: {total_loss/batch_num:.4f}")

    # validation loop
    model.eval()
    with torch.no_grad():
        loss = 0
        for batch in tqdm(train_dataloader):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            token_type_ids = batch['token_type_ids']
            labels = batch['labels']
            text = batch['text']
            # print(f"input_ids: {input_ids.shape} - {input_ids}")
            # print(f"attention_mask: {attention_mask.shape} - {attention_mask}")
            # print(f"token_type_ids: {token_type_ids.shape} - {token_type_ids}")
            # print(f"labels: {labels.shape} - {labels}")
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=labels)

            loss += outputs.loss
        print(f"EVAL LOSS: {loss}")
        scheduler.step(loss)
    model.train()

# print(tokenizer.decode(inputs['input_ids'].squeeze()))
