import argparse
import torch

datasets = [
    "Taskmaster/TM-1-2019/woz-dialogs.json",
    "Taskmaster/TM-1-2019/self-dialogs.json",
    "Taskmaster/TM-2-2020/data/flights.json",
    "Taskmaster/TM-2-2020/data/food-ordering.json",
    "Taskmaster/TM-2-2020/data/hotels.json",
    "Taskmaster/TM-2-2020/data/movies.json",
    "Taskmaster/TM-2-2020/data/music.json",
    "Taskmaster/TM-2-2020/data/restaurant-search.json",
    "Taskmaster/TM-2-2020/data/sports.json"
]


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("-bs", "--batch_size", type=int, default=10)
    parser.add_argument("--grad_accumulation_steps", type=int, default=5)
    parser.add_argument("-lr", '--learning_rate', type=float, default=1e-4)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pin_memory', action='store_false')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--testing_for_bugs', action='store_true')

    args = parser.parse_args()

    setattr(args, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    return vars(args)


def calculate_F1(TP, FP, FN):
    p = calculate_precision(TP, FP)
    r = calculate_recall(TP, FN)
    return 2*p*r/(p+r) if (p+r) > 0 else 0


def calculate_precision(TP, FP):
    return TP/(TP+FP) if (TP+FP) > 0 else 0


def calculate_recall(TP, FN):
    return TP/(TP+FN) if (TP+FN) > 0 else 0


def calculate_accuracy(TP, FP, FN, TN):
    return (TP+TN)/(TP+FP+FN+TN) if (TP+FP+FN+TN) > 0 else 0


def calculate_balanced_accuracy(TP, FP, FN, TN):
    r = calculate_recall(TP, FN)
    specificity = TN/(TN+FP)
    return (r+specificity)/2
