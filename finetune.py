import os
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast, GradScaler
from transformers import BertTokenizer, BertForTokenClassification
from dataset import load_taskmaster_datasets, load_multiwoz_dataset, VE_dataset, collate_class
from BertForValueExtraction import BertForValueExtraction
import utils
import argparse

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


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default="saved_models/bs30_gradacc20_lr1e-6_fp16-ACC0.8590")
    parser.add_argument('--finetuned_save_path', type=str, default="")
    parser.add_argument("-bs", "--batch_size", type=int, default=30)
    parser.add_argument("--grad_accumulation_steps", type=int, default=20)
    parser.add_argument("-lr", '--learning_rate', type=float, default=1e-6)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pin_memory', action='store_false')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--testing_for_bugs', action='store_true')
    parser.add_argument('--freeze_bert_layers', action='store_true')

    args = parser.parse_args()

    setattr(args, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    return vars(args)


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
    freeze_bert = kwargs['freeze_bert_layers']

    assert(kwargs['pretrained_model_path'] != ""), "specify a pre-trained model"
    assert(kwargs['finetuned_save_path'] != ""), "specify a save path for the fine-tuned model"
    # BertForValueExtraction
    assert(os.path.isdir(kwargs['pretrained_model_path']))
    from_pretrained = kwargs['pretrained_model_path']

    if fp16:
        scaler = GradScaler()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', model_max_length=128)  # for TM_1, out of 303066 samples, 5 are above 128 tokens

    train_data = load_multiwoz_dataset("multi-woz/train_dials.json", tokenizer, kwargs['testing_for_bugs'])
    val_data = load_multiwoz_dataset("multi-woz/dev_dials.json", tokenizer, kwargs['testing_for_bugs'])

    train_dataset = VE_dataset(train_data)
    val_dataset = VE_dataset(val_data)

    collator = collate_class(tokenizer.pad_token_id)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, num_workers=num_workers, pin_memory=pin_memory)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=int(batch_size/2), shuffle=True, collate_fn=collator, num_workers=num_workers, pin_memory=pin_memory)

    model = BertForValueExtraction(num_labels=len(label2id.keys()), from_pretrained=from_pretrained, freeze_bert=freeze_bert)
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
                model.save_(f"{kwargs['finetuned_save_path']}-ACC{best_acc:.4f}")

            else:
                count += 1

            if count == patience_limit:
                print("ran out of patience stopping early")
                break

        model.train()


if __name__ == "__main__":
    args = parse_args()

    if args['num_workers'] > 0:
        torch.multiprocessing.set_start_method("spawn")
    main(**args)
