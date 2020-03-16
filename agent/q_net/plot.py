import numpy as np
import matplotlib.pyplot as plt

g = np.load('GPTasPG_combined_r_rewards.npy')
#ls = np.load('GPTasPG_combined_r_length.npy')
#g = g / ls
avgs = []
for i in range(int(len(g) / 10)):
    running_g = g[i * 5: (i+1) * 5]
    avgs.append(np.mean(running_g))
avgs = np.array(avgs)

plt.plot([x*2 for x in range(len(avgs))],avgs)
plt.savefig('smoothed_full_rewards_gptpol.png')

