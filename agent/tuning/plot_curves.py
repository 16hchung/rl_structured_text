import numpy as np
import matplotlib.pyplot as plt

def plot_learning(losses_path, log_freq, save_path):
    losses = np.load(losses_path)
    epochs = np.arange(losses.size) 
    epochs *= log_freq
    plt.plot(epochs, losses, linewidth=1)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.savefig(save_path)
    plt.clf()

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--losses_path', type=str, default=None)
    parser.add_argument('--log_freq', type=int, default=2)
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args()

    plt.rcParams["font.family"] = "Times New Roman"

    if args.losses_path != None:
        plot_learning(args.losses_path, args.log_freq, args.save_path)
