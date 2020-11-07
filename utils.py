
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
