from transformers import BertForTokenClassification
import torch

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


class BertForValueExtraction(torch.nn.Module):
    def __init__(self, device, num_labels, from_pretrained='bert-base-uncased'):
        super(BertForValueExtraction, self).__init__()
        self.model = BertForTokenClassification.from_pretrained(from_pretrained,
                                                                num_labels=num_labels,
                                                                return_dict=True)
        self.device = device
        self.model.to(self.device)

    def calculate_loss(self, input_ids, attention_mask, token_type_ids, labels):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             labels=labels)
        return outputs.loss

    def predict(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids
                             )

        logits = outputs.logits
        preds = torch.max(logits, dim=2)[1]
        return preds

    def evaluate(self, preds, labels, attention_mask):
        TP, FP, FN, TN = 0, 0, 0, 0
        for pred, label, mask in zip(preds, labels, attention_mask):
            for p, l, m in zip(pred, label, mask):
                if m == 1:
                    if id2label[l] != "O":
                        if p == l:
                            TP += 1
                        else:
                            FN += 1
                    else:
                        if p == l:
                            TN += 1
                        else:
                            FP += 1
        return TP, FP, FN, TN

    def save_(self, model_path):
        self.model.save_pretrained(model_path)
