import sys
import os
import codecs
import numpy as np
import time

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
        if (predS[i] != '-'):
            for j in range(k, len(S_seq)):
                if (S_seq[j] != '-'):
                    k = j
                    break
            subset = T_seq[max(0,k-offset) : min(len(T_seq),k+offset+1)]
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
            subset = S_seq[max(0,k-offset) : min(len(S_seq),k+offset+1)]
            if (predS[i] in subset):
                count += 1
        k += 1
    NS = len(T_seq.replace('-', ''))
    ratio = count / NS
    return ratio

def DP(N1, N2, pair_score, S, T, Sgap, Tgap):
    DP = np.zeros((N1 + 1, N2 + 1))
    for i in range(1, N1 + 1):
        DP[i, 0] = DP[i - 1, 0] + Sgap[i - 1]
    for j in range(1, N2 + 1):
        DP[0][j] = DP[0, j - 1] + Tgap[j - 1]

    for i in range(1, N1 + 1):
        for j in range(1, N2 + 1):
            DP[i][j] = max(Tgap[j - 1] + DP[i, j - 1],
                           Sgap[i - 1] + DP[i - 1, j],
                           pair_score[i - 1, j - 1] + DP[i - 1, j - 1])

    i = N1
    j = N2
    compS = ""
    compT = ""
    while i > 0 and j > 0:
        if i > 0 and j > 0 and DP[i][j] == DP[i - 1][j - 1] + pair_score[i - 1][j - 1]:
            compS = S[i - 1] + compS
            compT = T[j - 1] + compT
            i -= 1
            j -= 1
        elif i > 0 and DP[i][j] == Sgap[i - 1] + DP[i - 1][j]:
            compS = S[i - 1] + compS
            compT = "-" + compT
            i -= 1
        elif j > 0 and DP[i][j] == Tgap[j - 1] + DP[i][j - 1]:
            compS = "-" + compS
            compT = T[j - 1] + compT
            j -= 1
    return compS, compT


def predict_via_dp(s_feat, t_feat, S_seq_ground_truth, T_seq_ground_truth):
    pair_score = np.zeros((s_feat.shape[0], t_feat.shape[0]))
    sgap = np.zeros(s_feat.shape[0])
    tgap = np.zeros(t_feat.shape[0])
    for i in range(s_feat.shape[0]):
        for j in range(t_feat.shape[0]):
            pair_score[i][j] = np.sum(s_feat[i])+  np.sum(t_feat[j])

    for i in range(s_feat.shape[0]):
        sgap[i] = np.sum(s_feat[i])
    for j in range(t_feat.shape[0]):
        tgap[j] = np.sum(t_feat[j])

    s_seq = ""
    for i in range(len(S_seq_ground_truth)):
        if S_seq_ground_truth[i] != '-':
            s_seq = s_seq + S_seq_ground_truth[i]

    t_seq = ""
    for i in range(len(T_seq_ground_truth)):
        if T_seq_ground_truth[i] != '-':
            t_seq = t_seq + T_seq_ground_truth[i]
    st=time.time()
    result =DP(s_feat.shape[0], t_feat.shape[0], pair_score, s_seq, t_seq, sgap, tgap)
    used=time.time() - st
    return result, used

if __name__ == '__main__':
    fastapath = os.path.join(sys.argv[1], "training_set")
    feature_path = os.path.join(sys.argv[1], "feature_set")
    print(fastapath)
    precision0 = []
    precision4 = []
    precision10=[]

    recall0 = []
    recall4 = []
    recall10= []
    area_loss = []
    test_size=10
    cnt=0
    used = []
    for root, dirs, files in os.walk(fastapath):
        for name in files:
            print(cnt,test_size,end='\r')
            sys.stdout.flush()
            fasta_file = os.path.join(root, name)
            content = codecs.open(fasta_file, 'r').read().split("\n")
            S_seq_name = content[0].strip()[1:]
            S_seq_ground_truth = content[1].strip()
            T_seq_name = content[2].strip()[1:]
            T_seq_ground_truth = content[3].strip()
            S_feat = np.loadtxt(os.path.join(feature_path, S_seq_name + ".txt"))
            T_feat = np.loadtxt(os.path.join(feature_path, T_seq_name + ".txt"))
            st = time.time()

            (predS, predT), theused=predict_via_dp(S_feat, T_feat, S_seq_ground_truth, T_seq_ground_truth)

            used.append(theused)
            precision0.append(precision(S_seq_ground_truth, T_seq_ground_truth, predS, predT, offset=0))
            precision4.append(precision(S_seq_ground_truth, T_seq_ground_truth, predS, predT, offset=4))

            precision10.append(precision(S_seq_ground_truth, T_seq_ground_truth, predS, predT, offset=10))
            recall0.append(recall(S_seq_ground_truth, T_seq_ground_truth, predS, predT, offset=0))
            recall4.append(recall(S_seq_ground_truth, T_seq_ground_truth, predS, predT, offset=4))
            recall10.append(recall(S_seq_ground_truth, T_seq_ground_truth, predS, predT, offset=10))
            area_loss.append(compute_area_unit(S_seq_ground_truth, T_seq_ground_truth, predS, predT))
            cnt+=1
            # print("precision0:\t", precision0[-1])
            # print("precision4:\t", precision4[-1])
            # print("recall0:\t", recall0[-1])
            # print("recall4:\t", recall4[-1])
            # print("area loss:\t", area_loss[-1])
            # print("-"*40)
            if cnt>=test_size:
                break

    # print
    # print("files:\t", files)
    print("\n AVERAGE:")
    # print("precision0:\t", sum(precision0) / test_size)
    # print("precision4:\t", sum(precision4) / test_size)
    # print("recall0:\t", sum(recall0) / test_size)
    # print("recall4:\t", sum(recall4) / test_size)
    # print("area loss:\t", sum(area_loss) / test_size)

    print(" {:.1f}\\% / {:.1f}\\% / {:.1f}\\% & {:.1f}\\% / {:.1f} \\% / {:.1f} \\%".format(np.average(precision0)*100,
                                                                            np.average(precision4)*100,
                                                                            np.average(precision10)*100,
                                                                            np.average(recall0)*100,
                                                                            np.average(recall4)*100,
                                                                            np.average(recall10)*100))
    used = np.asarray(used)
    # print("[", end="")
    for x in used:
        print(x)
    # print("]")
