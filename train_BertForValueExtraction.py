from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertForTokenClassification
from dataset import load_TM_1_data, TM_1_dataset, collate_class
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 5
gradient_accumulation_steps = 100
initial_lr = 1e-5
patience_limit = 5
num_workers = 4


def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', model_max_length=128)  # for TM_1, out of 303066 samples, 5 are above 128 tokens

    train_data, val_data = load_TM_1_data(tokenizer, for_testing_purposes=False, train_percent=0.9)

    train_dataset = TM_1_dataset(train_data)
    val_dataset = TM_1_dataset(val_data)

    collator = collate_class(tokenizer.pad_token_id, device=device)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, num_workers=num_workers, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, num_workers=num_workers, pin_memory=True)

    # BertForValueExtraction
    model = BertForValueExtraction(num_labels=len(label2id.keys()))
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=1, min_lr=initial_lr/100, verbose=True)

    best_acc, count = 0, 0

    for epoch in range(100):
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
            loss = model.calculate_loss(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        labels=labels)
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
            TP, FP, FN, TN = 0, 0, 0, 0
            for batch in tqdm(val_dataloader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['labels'].to(device)
                text = batch['text']

                preds = model.predict(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      token_type_ids=token_type_ids)

                tp, fp, fn, tn = model.evaluate(preds.tolist(), labels.tolist(), attention_mask.tolist())
                TP += tp
                FP += fp
                FN += fn
                TN += tn

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

    # print(tokenizer.decode(inputs['input_ids'].squeeze()))


if __name__ == "__main__":
    if num_workers > 0:
        torch.multiprocessing.set_start_method("spawn")
    main()
