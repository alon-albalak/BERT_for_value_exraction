import os
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast, GradScaler
from transformers import BertTokenizer, BertForTokenClassification
from dataset import load_taskmaster_datasets, taskmaster_dataset, collate_class
from BertForValueExtraction import BertForValueExtraction
import utils

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


def main(**kwargs):
    fp16 = kwargs['fp16']
    batch_size = kwargs['batch_size']
    num_workers = kwargs['num_workers']
    pin_memory = kwargs['pin_memory']
    patience_limit = kwargs['patience']
    initial_lr = kwargs['learning_rate']
    gradient_accumulation_steps = kwargs['grad_accumulation_steps']
    device = kwargs['device']
    epochs = kwargs['epochs']

    if fp16:
        scaler = GradScaler()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', model_max_length=128)  # for TM_1, out of 303066 samples, 5 are above 128 tokens

    train_data, val_data = load_taskmaster_datasets(utils.datasets, tokenizer, train_percent=0.9, for_testing_purposes=kwargs['testing_for_bugs'])

    train_dataset = taskmaster_dataset(train_data)
    val_dataset = taskmaster_dataset(val_data)

    collator = collate_class(tokenizer.pad_token_id)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, num_workers=num_workers, pin_memory=pin_memory)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=int(batch_size/2), shuffle=True, collate_fn=collator, num_workers=num_workers, pin_memory=pin_memory)

    # BertForValueExtraction
    if kwargs['model_path'] and os.path.isdir(kwargs['model_path']):
        from_pretrained = kwargs['model_path']
    else:
        from_pretrained = 'bert-base-uncased'
    model = BertForValueExtraction(num_labels=len(label2id.keys()), from_pretrained=from_pretrained)
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=1, min_lr=initial_lr/100, verbose=True)

    best_acc, count = 0, 0

    for epoch in range(epochs):
        # train loop
        total_loss = 0
        optimizer.zero_grad()
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, batch in pbar:

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            text = batch['text']
            # print(f"input_ids: {input_ids.shape} - {input_ids}")
            # print(f"attention_mask: {attention_mask.shape} - {attention_mask}")
            # print(f"token_type_ids: {token_type_ids.shape} - {token_type_ids}")
            # print(f"labels: {labels.shape} - {labels}")
            if fp16:
                with autocast():
                    loss = model.calculate_loss(input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                token_type_ids=token_type_ids,
                                                labels=labels)
                    total_loss += loss.item()
                    loss = loss/gradient_accumulation_steps

                scaler.scale(loss).backward()

            else:
                loss = model.calculate_loss(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            token_type_ids=token_type_ids,
                                            labels=labels)
                loss.backward()
                total_loss += loss.item()
            if ((i+1) % gradient_accumulation_steps) == 0:
                if fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            batch_num = ((i+1)/gradient_accumulation_steps)
            pbar.set_description(f"Loss: {total_loss/batch_num:.4f}")

        # validation loop
        model.eval()
        with torch.no_grad():
            TP, FP, FN, TN = model.evaluate(val_dataloader, device)

            pr = utils.calculate_precision(TP, FP)
            re = utils.calculate_recall(TP, FN)
            F1 = utils.calculate_F1(TP, FP, FN)
            acc = utils.calculate_accuracy(TP, FP, FN, TN)
            balanced_acc = utils.calculate_balanced_accuracy(TP, FP, FN, TN)
            print(f"Validation: pr {pr:.4f} - re {re:.4f} - F1 {F1:.4f} - acc {acc:.4f} - balanced acc {balanced_acc:.4f}")
            scheduler.step(balanced_acc)

            if balanced_acc > best_acc:
                best_acc = balanced_acc
                count = 0
            else:
                count += 1

            print(count)
            if count == patience_limit:
                print("ran out of patience stopping early")
                break

        model.train()

    if kwargs['model_path']:
        model.save_(kwargs['model_path'])

if __name__ == "__main__":
    args = utils.parse_args()

    if args['num_workers'] > 0:
        torch.multiprocessing.set_start_method("spawn")
    main(**args)
