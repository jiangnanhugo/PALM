import numpy as np
import string
import os

training_set_path = "/scratch/brown/ding274/df_align/training_set/"
feature_set_path = "/scratch/brown/ding274/df_align/feature_set/"


def precision(S_seq, T_seq, predS, predT, offset=0):
    count = 0
    k = 0
    for i in range(len(predS)):
        if (predS[i] != '-'):
            for j in range(k, len(S_seq)):
                if (S_seq[j] != '-'):
                    k = j
                    break
            subset = T_seq[max(0, k - offset): min(len(T_seq), k + offset + 1)]
            if (predT[i] in subset):
                count += 1
        k += 1
    NS = len(S_seq.replace('-', ''))
    ratio = count / NS
    return ratio


def recall(S_seq, T_seq, predS, predT, offset=0):
    count = 0
    k = 0
    for i in range(len(predT)):
        if (predT[i] != '-'):
            for j in range(k, len(T_seq)):
                if (T_seq[j] != '-'):
                    k = j
                    break
            subset = S_seq[max(0, k - offset): min(len(S_seq), k + offset + 1)]
            if (predS[i] in subset):
                count += 1
        k += 1
    NS = len(T_seq.replace('-', ''))
    ratio = count / NS
    return ratio


if __name__ == '__main__':
    test_size = 50
    n = 0
    files = []
    precision0 = []
    precision4 = []
    recall0 = []
    recall4 = []
    root = os.listdir(training_set_path)
    epoch = 10
    model_para = "paul_para_{%d}.npy".format(epoch)  # parameter saved path
    for file in root:
        if (n < test_size):
            n += 1
            files.append(file)
            with open(training_set_path + file, 'r') as f:
                S_name = f.readline()[1:].replace('\n', '').replace('\r', '')
                S_seq = f.readline().replace('\n', '').replace('\r', '')
                T_name = f.readline()[1:].replace('\n', '').replace('\r', '')
                T_seq = f.readline().replace('\n', '').replace('\r', '')
                S_train = S_seq.replace('-', '')
                T_train = T_seq.replace('-', '')

                print("S1: " + S_name + "\tS2: " + T_name)

                Paul = PAUL()
                Paul.load_para(model_para)  # load parameters
                Paul.read_sequence(S_name, T_name, S_train, T_train)
                predS, predT = Paul.inference()
                precision0.append(precision(S_seq, T_seq, predS, predT, offset=0))
                precision4.append(precision(S_seq, T_seq, predS, predT, offset=4))
                recall0.append(recall(S_seq, T_seq, predS, predT, offset=0))
                recall4.append(recall(S_seq, T_seq, predS, predT, offset=4))
    # print
    print("files:\t", files)
    print("precision0:\t", precision0)
    print("precision4:\t", precision4)
    print("recall0:\t", recall0)
    print("recall4:\t", recall4)
