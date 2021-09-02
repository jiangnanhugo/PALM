import numpy as np
import string
import sys
from palm import PALM
import os
import argparse


model_path = "./model/"

parser = argparse.ArgumentParser(description='PALM')
parser.add_argument('-m', '--model', type=str, help="mll or palm", default="palm")
parser.add_argument('-e', '--epoch', type=int, help="model epoch", default="10")
parser.add_argument('-t', '--testpath', type=str, help="path", default="")
parser.add_argument('-w', '--weight', type=float, help="tune the ratio between phi and area unit", default=100)
args = parser.parse_args()
epoch = args.epoch
model = args.model
if model =="mll":
    model_para = model_path + model + "_para_{}_w_0.npy".format(epoch)  # parameter saved path
else:
    model_para = model_path + model + "_para_{}_w_{}.npy".format(epoch, int(args.weight))  # parameter saved path
print(model_para)

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
    n = 0
    testing_set_path = args.testpath
    files = []
    precision0 = []
    precision4 = []
    precision10 = []
    recall0 = []
    recall4 = []
    recall10 = []
    # area_loss = []
    root = os.listdir(testing_set_path)
    test_size = 200
    Palm = PALM(model)

    Palm.load_para(model_para)  # load parameters
    for file in root:
        if (n < test_size):
            n += 1
            print("The [{}/{}] file".format(n, test_size), end="\r")
            sys.stdout.flush()
            files.append(file)
            with open(args.testpath + file, 'r') as f:
                S_name = f.readline()[1:].replace('\n', '').replace('\r', '')
                S_seq = f.readline().replace('\n', '').replace('\r', '')
                T_name = f.readline()[1:].replace('\n', '').replace('\r', '')
                T_seq = f.readline().replace('\n', '').replace('\r', '')
                S_train = S_seq.replace('-', '')
                T_train = T_seq.replace('-', '')

                # print("S1: " + S_name + "\tS2: " + T_name)
                Palm.read_sequence(S_name, T_name, S_train, T_train)
                predS, predT = Palm.test_inference()
                precision0.append(precision(S_seq, T_seq, predS, predT, offset=0))
                precision4.append(precision(S_seq, T_seq, predS, predT, offset=4))
                recall0.append(recall(S_seq, T_seq, predS, predT, offset=0))
                recall4.append(recall(S_seq, T_seq, predS, predT, offset=4))
                precision10.append(precision(S_seq, T_seq, predS, predT, offset=10))
                recall10.append(recall(S_seq, T_seq, predS, predT, offset=10))


    # print("files:\t", files)
    # print("\n AVERAGE:")
    print("{:.1f}\\% / {:.1f}\\% / {:.1f}\\% & {:.1f}\\% / {:.1f} \\% / {:.1f} \\%".format(np.average(precision0) * 100,
                                                                                           np.average(precision4) * 100,
                                                                                           np.average(precision10) * 100,
                                                                                           np.average(recall0) * 100,
                                                                                           np.average(recall4) * 100,
                                                                                           np.average(recall10) * 100))
