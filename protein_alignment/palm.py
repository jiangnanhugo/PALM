import numpy as np


feature_set_path = "./feature_set/"
model_path = "./model/"


class PALM(object):
    def __init__(self, model):
        self.dimension = 82
        self.model = model
        np.random.seed(0)
        self.theta = np.random.normal(0, 1, self.dimension)

    def extract_feature(self, name):
        ls = []
        with open(feature_set_path + name + ".txt", 'r') as f:
            for line in f:
                line = line.strip('\n').strip()
                ls.append(line.split(' '))
        ls = np.array(ls, dtype=float)
        return ls

    def load_para(self, model_para):
        self.theta = np.load(model_para)

    def save_para(self, epoch, weight):
        np.save(model_path + self.model + "_para_{}_w_{}.npy".format(epoch, weight), self.theta)

    def read_sequence(self, S_name, T_name, S_train, T_train):
        S_feature = self.extract_feature(S_name)
        T_feature = self.extract_feature(T_name)
        self.S_train = S_train
        self.T_train = T_train
        self.NS = len(S_train)
        self.NT = len(T_train)
        # print('NS:\t{}\t NT:\t{}'.format(self.NS, self.NT))

        ## ensure lengths are equal
        if (len(S_feature) < self.NS):
            tmp = np.ones((self.NS - len(S_feature), S_feature[0].size))
            S_feature = np.vstack((S_feature, tmp))
        if (len(T_feature) < self.NT):
            tmp = np.ones((self.NT - len(T_feature), T_feature[0].size))
            T_feature = np.vstack((T_feature, tmp))

        self.phi = np.zeros((self.NS + 1, self.NT + 1, 3, self.dimension))  # M Is It
        self.score = np.zeros((self.NS + 1, self.NT + 1, 3))  # M Is It
        for i in range(self.NS):
            for j in range(self.NT):
                self.phi[i, j, 0] = np.concatenate((S_feature[i] / 41.0, T_feature[j] / 41.0), axis=0)
                self.phi[i, j, 1] = np.concatenate((S_feature[i] / 41.0, np.zeros(41)), axis=0)
                self.phi[i, j, 2] = np.concatenate((np.zeros(41), T_feature[j] / 41.0), axis=0)
                self.score[i, j, 0] = np.matmul(self.theta, self.phi[i][j][0])  # log_scale
                self.score[i, j, 1] = np.matmul(self.theta, self.phi[i][j][1])
                self.score[i, j, 2] = np.matmul(self.theta, self.phi[i][j][2])
        for i in range(self.NS):
            self.phi[i, self.NT, 1] = np.concatenate((S_feature[i], np.zeros(41)), axis=0)
            self.score[i, self.NT, 1] = np.matmul(self.theta, self.phi[i][self.NT][1])  # log scale
        for j in range(self.NT):
            self.phi[self.NS, j, 2] = np.concatenate((np.zeros(41), T_feature[j] / 41.0), axis=0)
            self.score[self.NS, j, 2] = np.matmul(self.theta, self.phi[self.NS][j][2])  # log scale
        return

    def backward_Z(self):
        Z = np.zeros((self.NS + 1, self.NT + 1))
        for i in range(self.NS - 1, -1, -1):
            Z[i][self.NT] = Z[i + 1][self.NT] + self.score[i][self.NT][1]
        for j in range(self.NT - 1, -1, -1):
            Z[self.NS][j] = Z[self.NS][j + 1] + self.score[self.NS][j][2]
        for i in range(self.NS - 1, -1, -1):
            for j in range(self.NT - 1, -1, -1):
                C = max(Z[i + 1][j + 1] + self.score[i][j][0], Z[i + 1][j] + self.score[i][j][1],
                        Z[i][j + 1] + self.score[i][j][2])
                tmp = np.exp(Z[i + 1][j + 1] + self.score[i][j][0] - C) + np.exp(
                    Z[i + 1][j] + self.score[i][j][1] - C) + np.exp(Z[i][j + 1] + self.score[i][j][2] - C)
                Z[i, j] = C + np.log(tmp)
        self.Z = Z
        return

    def test_inference(self):
        A = np.zeros((self.NS + 1, self.NT + 1))
        for i in range(1, self.NS + 1):
            A[i][0] = A[i - 1][0] + self.score[i - 1][0][1]
        for j in range(1, self.NT + 1):
            A[0][j] = A[0][j - 1] + self.score[0][j - 1][2]
        for i in range(1, self.NS + 1):
            for j in range(1, self.NT + 1):
                A[i][j] = max(A[i - 1][j - 1] + self.score[i - 1][j - 1][0], A[i - 1][j] + self.score[i - 1][j][1],
                              A[i][j - 1] + self.score[i][j - 1][2])

        # from matrix A to find the alignment
        self.predS = ""
        self.predT = ""
        i = self.NS
        j = self.NT
        while (i > 0 or j > 0):
            if (i > 0 and j > 0 and A[i][j] == A[i - 1][j - 1] + self.score[i - 1][j - 1][0]):
                self.predS = self.S_train[i - 1] + self.predS
                self.predT = self.T_train[j - 1] + self.predT
                i -= 1
                j -= 1
            elif (i > 0 and A[i][j] == A[i - 1][j] + self.score[i - 1][j][1]):
                self.predS = self.S_train[i - 1] + self.predS
                self.predT = "-" + self.predT
                i -= 1
            elif (j > 0 and A[i][j] == A[i][j - 1] + self.score[i][j - 1][2]):
                self.predS = "-" + self.predS
                self.predT = self.T_train[j - 1] + self.predT
                j -= 1
            else:
                print("inference ERROR in position (%d, %d)" % (i, j))
                exit(0)
        return self.predS, self.predT

    def al_x_from_J(self, S, T, Y):
        x = 0
        y = 0
        for al in range(len(T)):
            if (T[al] != '-'):
                y += 1
            if (S[al] != '-'):
                x += 1
            if (y == Y):
                return al, x

    def train_inference(self, S_seq, T_seq, weight=1.0):
        if (self.model == "mll"):
            self.predS = S_seq
            self.predT = T_seq
        elif (self.model == "palm"):
            self.predS = ""
            self.predT = ""
            A = np.zeros((self.NS + 1, self.NT + 1))
            for i in range(self.NS - 1, -1, -1):
                A[i][self.NT] = A[i + 1][self.NT] + self.score[i][self.NT][1]
            for j in range(self.NT - 1, -1, -1):
                al, x = self.al_x_from_J(S_seq, T_seq, j + 1)
                if (S_seq[al] == '-'):
                    A[self.NS][j] = A[self.NS][j + 1] + self.score[self.NS][j][2] - np.abs(x - self.NS) * weight
                else:
                    A[self.NS][j] = A[self.NS][j + 1] + self.score[self.NS][j][2] - np.abs(x - self.NS + 0.5) * weight
            for i in range(self.NS - 1, -1, -1):
                for j in range(self.NT - 1, -1, -1):
                    al, x = self.al_x_from_J(S_seq, T_seq, j + 1)
                    if (S_seq[al] == '-'):
                        A[i][j] = max(A[i + 1][j + 1] + self.score[i][j][0] - np.abs(x - i - 0.5) * weight,
                                      A[i + 1][j] + self.score[i][j][1],
                                      A[i][j + 1] + self.score[i][j][2] - np.abs(x - i) * weight)
                    else:
                        A[i][j] = max(A[i + 1][j + 1] + self.score[i][j][0] - np.abs(x - i) * weight,
                                      A[i + 1][j] + self.score[i][j][1],
                                      A[i][j + 1] + self.score[i][j][2] - np.abs(x - i + 0.5) * weight)

            # from matrix A to find the alignment
            i = 0
            j = 0
            while (i < self.NS or j < self.NT):
                if (i < self.NS and j < self.NT and A[i][j]):
                    al, x = self.al_x_from_J(S_seq, T_seq, j + 1)
                    if (S_seq[al] == '-'):
                        if (A[i][j] == A[i + 1][j + 1] + self.score[i][j][0] - np.abs(x - i - 0.5) * weight):
                            self.predS += self.S_train[i]
                            self.predT += self.T_train[j]
                            i += 1
                            j += 1
                        elif (A[i][j] == A[i + 1][j] + self.score[i][j][1]):
                            self.predS += self.S_train[i]
                            self.predT += "-"
                            i += 1
                        elif (A[i][j] == A[i][j + 1] + self.score[i][j][2] - np.abs(x - i) * weight):
                            self.predS += "-"
                            self.predT += self.T_train[j]
                            j += 1
                        else:
                            print("inference ERROR in position (%d, %d)" % (i, j))
                            exit(0)
                    else:
                        if (A[i][j] == A[i + 1][j + 1] + self.score[i][j][0] - np.abs(x - i) * weight):
                            self.predS += self.S_train[i]
                            self.predT += self.T_train[j]
                            i += 1
                            j += 1
                        elif (A[i][j] == A[i + 1][j] + self.score[i][j][1]):
                            self.predS += self.S_train[i]
                            self.predT += "-"
                            i += 1
                        elif (A[i][j] == A[i][j + 1] + self.score[i][j][2] - np.abs(x - i + 0.5) * weight):
                            self.predS += "-"
                            self.predT += self.T_train[j]
                            j += 1
                        else:
                            print("inference ERROR in position (%d, %d)" % (i, j))
                            exit(0)
                elif (i == self.NS):
                    self.predS += '-' * len(self.T_train[j:])
                    self.predT += self.T_train[j:]
                    break
                elif (j == self.NT):
                    self.predS += self.S_train[i:]
                    self.predT += '-' * len(self.S_train[i:])
                    break
                else:
                    print("ERROR!")
        else:
            print("ERROR! model not specified!:\t" + self.model)
            exit(0)
        return

    def compute_area_unit(self, S_seq, T_seq):
        self.L_area_unit = []
        x_gt = 0
        y_gt = 0
        x_pred = 0
        y_pred = 0
        tmp = 0
        for al_gt in range(len(T_seq)):
            if (T_seq[al_gt] == '-'):
                x_gt += 1
            elif (S_seq[al_gt] == '-'):
                for al_pred in range(tmp, len(self.predT)):
                    if (self.predT[al_pred] == '-'):
                        x_pred += 1
                    elif (self.predS[al_pred] == '-'):
                        self.L_area_unit.append(np.abs(x_gt - x_pred))
                        y_pred += 1
                        tmp = al_pred + 1
                        break
                    else:
                        self.L_area_unit.append(np.abs(x_gt - x_pred - 0.5))
                        x_pred += 1
                        y_pred += 1
                        tmp = al_pred + 1
                        break
                y_gt += 1
            else:
                for al_pred in range(tmp, len(self.predT)):
                    if (self.predT[al_pred] == '-'):
                        x_pred += 1
                    elif (self.predS[al_pred] == '-'):
                        self.L_area_unit.append(np.abs(x_gt - x_pred + 0.5))
                        y_pred += 1
                        tmp = al_pred + 1
                        break
                    else:
                        self.L_area_unit.append(np.abs(x_gt - x_pred))
                        x_pred += 1
                        y_pred += 1
                        tmp = al_pred + 1
                        break
                x_gt += 1
                y_gt += 1
        return np.sum(np.array(self.L_area_unit))

    def compute_gradient(self, Sample_size):
        self.gradient = 0
        x_gt = 0
        y_gt = 0
        grad_1 = 0
        grad_log_Z = 0
        self.backward_Z()
        for al_gt in range(len(self.predT)):

            if (self.predT[al_gt] == '-'):
                grad_1 += self.phi[x_gt, y_gt, 1]
                x_gt += 1
            elif (self.predS[al_gt] == '-'):
                grad_1 += self.phi[x_gt, y_gt, 2]
                y_gt += 1
            else:
                grad_1 += self.phi[x_gt, y_gt, 0]
                x_gt += 1
                y_gt += 1
        Loss1 = np.matmul(self.theta, grad_1)
        Loss = Loss1 - self.Z[0, 0]
        for m in range(Sample_size):
            grad_log_Z += self.forward_sampling()
        self.gradient = grad_1 - grad_log_Z / Sample_size
        return Loss

    def forward_sampling(self):
        grad = 0
        i = 0
        j = 0
        while (i < self.NS or j < self.NT):
            if (i == self.NS):
                grad += self.phi[i][j][2]
                j += 1
            elif (j == self.NT):
                grad += self.phi[i][j][1]
                i += 1
            else:
                P_M = np.exp(self.score[i][j][0] + self.Z[i + 1, j + 1] - self.Z[i, j])
                P_Is = np.exp(self.score[i][j][1] + self.Z[i + 1, j] - self.Z[i, j])
                P_It = np.exp(self.score[i][j][2] + self.Z[i, j + 1] - self.Z[i, j])
                ZP = np.sum([P_M, P_Is, P_It])
                choice = np.random.multinomial(1, [P_M / ZP, P_Is / ZP, P_It / ZP])

                idx = np.where(choice == 1)[0][0]
                if (idx == 0):
                    grad += self.phi[i][j][0]
                    i += 1
                    j += 1
                elif (idx == 1):
                    grad += self.phi[i][j][1]
                    i += 1
                elif (idx == 2):
                    grad += self.phi[i][j][2]
                    j += 1
                else:
                    print("sampling ERROR in position (%d, %d)" % (i, j))

        return grad

    def update_theta(self, lamda):  # gradient update
        self.theta = self.theta + lamda * self.gradient
        return
