import torch
import numpy as np
from sklearn.model_selection import train_test_split
import random
import os



def load_from_dir(root_dir):
    """
    root_dir: train / val / test
    return:
        X: (N, 2, 2048)
        Y: (N,)
    """
    X, Y = [], []

    dev_folders = sorted(os.listdir(root_dir))
    for dev in dev_folders:
        label = int(dev.replace("dev", ""))  # dev03 → 3
        dev_path = os.path.join(root_dir, dev)

        for f in os.listdir(dev_path):
            if f.endswith(".npy"):
                x = np.load(os.path.join(dev_path, f))
                X.append(x)
                Y.append(label)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.uint8)
    return X, Y

def TrainDataset(num, rand_num):
    #x = np.load(f"/data/fuxue/Dataset_4800/X_train_{num}Class.npy")
    #y = np.load(f"/data/fuxue/Dataset_4800/Y_train_{num}Class.npy")
    x, y = load_from_dir("C:/kaiji/train")
    y = y.astype(np.uint8)
    X_train_labeled1, X_train_unlabeled1, Y_train_labeled1, Y_train_unlabeled1 = train_test_split(x, y, test_size=0.5, random_state=rand_num)
    X_train_labeled2, X_train_unlabeled2, Y_train_labeled2, Y_train_unlabeled2 = train_test_split(X_train_labeled1,Y_train_labeled1, test_size=0.6, random_state=rand_num)
    X_train_labeled3, X_train_unlabeled3, Y_train_labeled3, Y_train_unlabeled3 = train_test_split(X_train_labeled2,Y_train_labeled2,test_size=0.5,random_state=rand_num)

    X_train_label, X_val, Y_train_label, Y_val = train_test_split(X_train_labeled3, Y_train_labeled3, test_size=0.3, random_state=rand_num)

    X_train_unlabeled = np.concatenate((X_train_unlabeled1,X_train_unlabeled2,X_train_unlabeled3), axis=0)
    Y_train_unlabeled = np.concatenate((Y_train_unlabeled1, Y_train_unlabeled2,Y_train_unlabeled3), axis=0)

    return X_train_label, X_train_unlabeled, x, X_val, Y_train_label, Y_train_unlabeled, y, Y_val

def TestDataset(num):
    #x = np.load(f"/data/fuxue/Dataset_4800/X_test_{num}Class.npy")
    #y = np.load(f"/data/fuxue/Dataset_4800/Y_test_{num}Class.npy")
    x, y = load_from_dir("C:/kaiji/test")
    #y = y.astype(np.uint8)
    return x, y
