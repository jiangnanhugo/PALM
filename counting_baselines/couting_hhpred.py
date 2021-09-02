import numpy as np
import os, sys
import codecs

from shutil import copyfile


def compute_area_unit(S_seq, T_seq, predS, predT):
    L_area_unit = []
    x_gt = 0
    y_gt = 0
    x_pred = 0
    y_pred = 0
    tmp = 0
    for al_gt in range(len(T_seq)):
        if (T_seq[al_gt] == '-'):
            x_gt += 1
        elif (S_seq[al_gt] == '-'):
            for al_pred in range(tmp, len(predT)):
                if (predT[al_pred] == '-'):
                    x_pred += 1
                elif (predS[al_pred] == '-'):
                    L_area_unit.append(np.abs(x_gt - x_pred))
                    y_pred += 1
                    tmp = al_pred + 1
                    break
                else:
                    L_area_unit.append(np.abs(x_gt - x_pred - 0.5))
                    x_pred += 1
                    y_pred += 1
                    tmp = al_pred + 1
                    break
            y_gt += 1
        else:
            for al_pred in range(tmp, len(predT)):
                if (predT[al_pred] == '-'):
                    x_pred += 1
                elif (predS[al_pred] == '-'):
                    L_area_unit.append(np.abs(x_gt - x_pred + 0.5))
                    y_pred += 1
                    tmp = al_pred + 1
                    break
                else:
                    L_area_unit.append(np.abs(x_gt - x_pred))
                    x_pred += 1
                    y_pred += 1
                    tmp = al_pred + 1
                    break
            x_gt += 1
            y_gt += 1
    # print(self.L_area_unit)
    return np.sum(np.array(L_area_unit))


def precision(S_seq, T_seq, predS, predT, offset=0):
    count = 0
    k = 0
    for i in range(len(predS)):
        if predS[i] != '-':
            for j in range(k, len(S_seq)):
                if S_seq[j] != '-':
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
        if predT[i] != '-':
            for j in range(k, len(T_seq)):
                if T_seq[j] != '-':
                    k = j
                    break
            subset = S_seq[max(0, k - offset): min(len(S_seq), k + offset + 1)]
            if (predS[i] in subset):
                count += 1
        k += 1
    NS = len(T_seq.replace('-', ''))
    ratio = count / NS
    return ratio


def hhpred_combined(base_dir,hhpred_dir_global, hhpred_dir_local, S_range, T_range):
    idential_fasta_files = []
    for root, dirs, files in os.walk(hhpred_dir_global):
        for filename in files:
            if filename.endswith(".fasta"):
                idential_fasta_files.append(filename)

    # print(len(d))
    precision0 = []
    precision4 = []
    recall0 = []
    recall4 = []
    area_loss = []
    for d in idential_fasta_files:
        if not os.path.isfile(os.path.join(base_dir, d)):
            continue
        ground_truth_fr = codecs.open(os.path.join(base_dir, d), 'r').read().split("\n")
        S_seq = ground_truth_fr[1]
        T_seq = ground_truth_fr[3]
        if S_range[0] <= len(S_seq) <= S_range[1] and T_range[0] <= len(T_seq) <= T_range[1]:
            hhpred_file_fr = codecs.open(os.path.join(hhpred_dir_global, d), 'r').read().split("\n")
            predS = hhpred_file_fr[1]
            predT = hhpred_file_fr[3]

            precision0.append(precision(S_seq, T_seq, predS, predT, offset=4))
            precision4.append(precision(S_seq, T_seq, predS, predT, offset=10))
            recall0.append(recall(S_seq, T_seq, predS, predT, offset=4))
            recall4.append(recall(S_seq, T_seq, predS, predT, offset=10))
            area_loss.append(compute_area_unit(S_seq, T_seq, predS, predT))
        else:
            continue

    idential_fasta_files = []
    for root, dirs, files in os.walk(hhpred_dir_local):
        for filename in files:
            if filename.endswith(".fasta"):
                idential_fasta_files.append(filename)
    for d in idential_fasta_files:
        if not os.path.isfile(os.path.join(base_dir, d)):
            continue
        ground_truth_fr = codecs.open(os.path.join(base_dir, d), 'r').read().split("\n")
        S_seq = ground_truth_fr[1]
        T_seq = ground_truth_fr[3]
        if S_range[0] <= len(S_seq) <= S_range[1] and T_range[0] <= len(T_seq) <= T_range[1]:
            hhpred_file_fr = codecs.open(os.path.join(hhpred_dir_local, d), 'r').read().split("\n")
            predS = hhpred_file_fr[1]
            predT = hhpred_file_fr[3]

            precision0.append(precision(S_seq, T_seq, predS, predT, offset=4))
            precision4.append(precision(S_seq, T_seq, predS, predT, offset=10))
            recall0.append(recall(S_seq, T_seq, predS, predT, offset=4))
            recall4.append(recall(S_seq, T_seq, predS, predT, offset=10))
            area_loss.append(compute_area_unit(S_seq, T_seq, predS, predT))
        else:
            continue

        # print(hhpred_predicted_S)

    print(" {:.1f}\\% & {:.1f}\\% & {:.1f}\\% & {:.1f}\\% & {:.0f} ".format(np.average(precision0) * 100,
                                                                            np.average(precision4) * 100,
                                                                            np.average(recall0) * 100,
                                                                            np.average(recall4) * 100,
                                                                            sum(area_loss) / (0.01 + len(area_loss))))


# counting for all the files where seq is less than 200
def get_training_set_names(base_dir, hhpred_dir, S_range, T_range):
    idential_fasta_files = []
    for root, dirs, files in os.walk(hhpred_dir):
        for filename in files:
            if filename.endswith(".fasta"):
                idential_fasta_files.append(filename)
    # print(len(d))
    precision0 = []
    precision4 = []
    recall0 = []
    recall4 = []
    area_loss = []
    for d in idential_fasta_files:
        if not os.path.isfile(os.path.join(base_dir, d)):
            continue
        ground_truth_fr = codecs.open(os.path.join(base_dir, d), 'r').read().split("\n")
        S_seq = ground_truth_fr[1]
        T_seq = ground_truth_fr[3]
        if S_range[0] <= len(S_seq) <= S_range[1] and T_range[0] <= len(T_seq) <= T_range[1]:
            hhpred_file_fr = codecs.open(os.path.join(hhpred_dir, d), 'r').read().split("\n")
            predS = hhpred_file_fr[1]
            predT = hhpred_file_fr[3]

            precision0.append(precision(S_seq, T_seq, predS, predT, offset=4))
            precision4.append(precision(S_seq, T_seq, predS, predT, offset=10))
            recall0.append(recall(S_seq, T_seq, predS, predT, offset=4))
            recall4.append(recall(S_seq, T_seq, predS, predT, offset=10))
            area_loss.append(compute_area_unit(S_seq, T_seq, predS, predT))
        else:
            continue

        # print(hhpred_predicted_S)

    print(" {:.1f}\\% & {:.1f}\\% & {:.1f}\\% & {:.1f}\\% & {:.0f} ".format(np.average(precision0) * 100,
                                                                            np.average(precision4) * 100,
                                                                            np.average(recall0) * 100,
                                                                            np.average(recall4) * 100,
                                                                            sum(area_loss) / (0.01 + len(area_loss))))


if __name__ == '__main__':
    training_set_dir = '/media/jiangnanhugo/Data/align_data/Alignment_DeepAlign/'
    S_ranges = [[0, 100], [100, 200], [0, 200], [0, 100],   [200, 400], [0, 400], [0, 200], [400, 1000000]]
    T_ranges = [[0, 100], [100, 200], [200, 400],[200, 400], [0, 200],  [400, 1000000], [200, 400], [400, 1000000]]
    hhpred_dir_global = "/media/jiangnanhugo/Data/align_data/Alignment_Test/inhouse/alignment/HHpred/HHpred_glob/"
    hhpred_dir_local = "/media/jiangnanhugo/Data/align_data/Alignment_Test/inhouse/alignment/HHpred/HHpred_loc/"
    print("--------Combined--------")
    for sran, tran in zip(S_ranges, T_ranges):
        print("S={} T={}".format(sran, tran), end="\t")
        hhpred_combined(training_set_dir,hhpred_dir_global, hhpred_dir_local, sran, tran)


    hhpred_dir = "/media/jiangnanhugo/Data/align_data/Alignment_Test/inhouse/alignment/CNF"
    print("--------CNF--------")
    for sran, tran in zip(S_ranges, T_ranges):
        print("S={} T={}".format(sran, tran), end="\t")
        get_training_set_names(training_set_dir, hhpred_dir, sran, tran)

    hhpred_dir = "/media/jiangnanhugo/Data/align_data/Alignment_Test/inhouse/alignment/DRNF/DRNF_DA_000"
    print("--------DRNF--------")
    for sran, tran in zip(S_ranges, T_ranges):
        print("S={} T={}".format(sran, tran), end="\t")
        get_training_set_names(training_set_dir, hhpred_dir, sran, tran)
