import numpy as np
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--infile', default='GPTasPG_combined_r_rewards.npy')
parser.add_argument('--outfile', default='smoothed_full_rewards_gptpol.png')
args = parser.parse_args()

g = np.load(args.infile)
#ls = np.load('GPTasPG_combined_r_length.npy')
#g = g / ls
avgs = []
for i in range(int(len(g) / 10)):
    running_g = g[i * 5: (i+1) * 5]
    avgs.append(np.mean(running_g))
avgs = np.array(avgs)

plt.plot([x*2 for x in range(len(avgs))],avgs)
plt.savefig(args.outfile)

