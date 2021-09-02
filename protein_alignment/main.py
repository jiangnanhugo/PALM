import autograd.numpy as np
import argparse
import os
from palm import PALM

NUM_EPOCHS = int(1000)
NUM_SAMPLES = 10
lamda = 1

training_set_path = "./training_set/"
feature_set_path = "./feature_set/"

parser = argparse.ArgumentParser(description='PALM')
parser.add_argument('-m', '--model', type=str, help="mll or palm", default="palm")
parser.add_argument('-w', '--weight', type=int, help="tune the ratio between phi and area unit", default=0)
args = parser.parse_args()
model = args.model

if __name__ == '__main__':
	Palm = PALM(model)
	root = os.listdir(training_set_path)
	Training_size = len(root)

	for epoch in range(NUM_EPOCHS):
		print('epoch [{}/{}]'.format(epoch, NUM_EPOCHS), flush=True)
		idx = np.random.choice(range(0, Training_size), 1, replace=False)[0]
		filename = root[idx]
		print("filename:\t" + filename + "\n", flush=True)
		with open(training_set_path + filename, 'r') as f:
			S_name = f.readline()[1:].replace('\n', '').replace('\r', '')
			S_seq = f.readline().replace('\n', '').replace('\r', '')
			T_name = f.readline()[1:].replace('\n', '').replace('\r', '')
			T_seq = f.readline().replace('\n', '').replace('\r', '')
		S_train = S_seq.replace('-', '')
		T_train = T_seq.replace('-', '')

		# save parameters every 10 epochs
		if epoch % 50 == 0:
			Palm.save_para(epoch, args.weight)

		### training
		# learning rate: lamda
		if epoch % 50 == 0:
			lamda = lamda *0.9

		Palm.read_sequence(S_name, T_name, S_train, T_train)
		Palm.train_inference(S_seq, T_seq, args.weight)
		print("S_seq:{}\npredS:{}\n".format(S_seq,Palm.predS))
		print("T_seq:{}\npredT:{}\n".format(T_seq, Palm.predT))
		loss = Palm.compute_gradient(NUM_SAMPLES)
		print("Loss:\t %.5f" % (loss))
		Palm.update_theta(lamda)


