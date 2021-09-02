#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include<iterator>
#include <algorithm>
#include <dirent.h>
#include <cmath>
#include "../protein_alignment-PAUL/align/seq.h"

using namespace std;
double threshold = 0.1;



void
Dp(int N1, int N2, string S, string T, vector <vector<double>> pair_score, vector<double> Sgap, vector<double> Tgap) {
    vector<double> tmp(N2 + 1, 0);
    vector <vector<double>> DP(N1 + 1, tmp);
    for (int i = 1; i <= N1; i++)
        DP[i][0] = DP[i - 1][0] + Sgap[i - 1];

    for (int j = 1; j <= N2; j++)
        DP[0][j] = DP[0][j - 1] + Tgap[j - 1];


    for (int i = 1; i <= N1; i++) {
        for (int j = 1; j <= N2; j++) {
            double tmp = max(Tgap[j - 1] + DP[i][j - 1],
                             Sgap[i - 1] + DP[i - 1][j]);
            DP[i][j] = max(pair_score[i - 1][j - 1] + DP[i - 1][j - 1],
                           tmp);
        }
    }
    // from DP to find the alignment
    string compS = "", compT = "";
    for (int i = N1, j = N2; i > 0 || j > 0;) {
        if (i > 0 && j > 0 && DP[i][j] == DP[i - 1][j - 1] + pair_score[i - 1][j - 1]) {
            compS = S[i - 1] + compS;
            compT = T[j - 1] + compT;
            i--, j--;
        } else if (i > 0 && DP[i][j] == Sgap[i - 1] + DP[i - 1][j]) {
            compS = S[i - 1] + compS;
            compT = "-" + compT;
            i--;
        } else if (j > 0 && DP[i][j] == Tgap[j - 1] + DP[i][j - 1]) {
            compS = "-" + compS;
            compT = T[j - 1] + compT;
            j--;
        } else {
            cout << "ERROR!\t i=" << i << "; j=" << j << endl;
        }
    }
    cout << "print alignment:\n" << compS << endl << compT << endl;
}

void calc_prec_and_recall(string S_seqname, string T_seqname, string TGTPath) {
    SEQUENCE *S_Sequence = new SEQUENCE(S_seqname, TGTPath, 0, 0);
    SEQUENCE *T_Sequence = new SEQUENCE(T_seqname, TGTPath, 0, 0);
    int N1 = S_Sequence->sequence.size(), N2 = T_Sequence->sequence.size();
    vector<double> tmp(N2, 1);
    vector <vector<double>> pair_score(N1, tmp);
    vector<double> Sgap(N1, 0);
    vector<double> Tgap(N2, 0);


    for (int i = 0; i < (int) S_Sequence->sequence.size(); i++) {
        for (int j = 0; j < (int) T_Sequence->sequence.size(); j++) {
            for (int t = 0; t < 8; t++) {
                pair_score[i][j] += S_Sequence->SS8[i][t] * T_Sequence->SS8[j][t];
                Sgap[i]+=S_Sequence->SS8[i][t];
                Tgap[j]+=T_Sequence->SS8[j][t];
            }
            for (int t = 0; t < 3; t++) {
                pair_score[i][j] += S_Sequence->acc_our_10_42[i][t] * T_Sequence->acc_our_10_42[j][t];
                Sgap[i]+=S_Sequence->SS8[i][t];
                Tgap[j]+=T_Sequence->SS8[j][t];
            }
            for (int t = 0; t < 20; t++) {
                pair_score[i][j] += S_Sequence->EmissionScore[i][t] * T_Sequence->EmissionScore[j][t];
                Sgap[i]+=S_Sequence->SS8[i][t];
                Tgap[j]+=T_Sequence->SS8[j][t];
            }
            for (int t = 0; t < 10; t++) {
                pair_score[i][j] += S_Sequence->ProfHMM[i][t] * T_Sequence->ProfHMM[j][t];
                Sgap[i]+=S_Sequence->SS8[i][t];
                Tgap[j]+=T_Sequence->SS8[j][t];
            }
        }
    }
    cout << "INPUT to DP:\n" << S_Sequence->sequence << endl << T_Sequence->sequence << endl;
    cout << N1 << " " << N2 << endl;
    Dp(N1, N2, S_Sequence->sequence, T_Sequence->sequence, pair_score, Sgap, Tgap);
}


void read_directory(const std::string &name, vector <string> &v) {
    DIR *dirp = opendir(name.c_str());
    struct dirent *dp;
    while ((dp = readdir(dirp)) != NULL) {
        v.push_back(dp->d_name);
    }
    closedir(dirp);
}

bool endswith(std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

int main(int argc, char **argv) {
    string FASTAPath = argv[1];
    string TGTPath = "/home/jiangnanhugo/Data/align_data/TGT_BC100/";
    vector <string> v;
    read_directory(FASTAPath, v);
    for (auto &element : v) {
        if (!endswith(element, ".fasta")) {
            continue;
        }
        cout << FASTAPath + element << endl;
        ifstream fasta_fr(FASTAPath + element);
        string S_seq_name, T_seq_name, S_seq_ground_truth, T_seq_ground_truth;
        if (fasta_fr.is_open()) {
            getline(fasta_fr, S_seq_name);
            getline(fasta_fr, S_seq_ground_truth);
            getline(fasta_fr, T_seq_name);
            getline(fasta_fr, T_seq_ground_truth);
            fasta_fr.close();
            S_seq_name = S_seq_name.substr(1);
            T_seq_name = T_seq_name.substr(1);
            calc_prec_and_recall(S_seq_name, T_seq_name, TGTPath);
        }
    }


}