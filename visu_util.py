# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_pcd_three_views(filename, pcds, titles, suptitle='', sizes=None, cmap='Set3', zdir='y',
                         xlim=(-0.4, 0.4), ylim=(-0.4, 0.4), zlim=(-0.3, 0.3)):
    if sizes is None:
        sizes = [0.5 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 3, 9))
    for i in range(3):
        elev = 30
        azim = -45 + 90 * i
        for ij, (pcd, size) in enumerate(zip(pcds, sizes)):
            if np.shape(pcd)[1] == 3:
                color = pcd[:, 0]
            else:
                if ij == 0 or ij == 3:
                    color = pcd[:, 3] - 0.5/11
                else:
                    color = (np.argmax(pcd[:, 3:], -1) + 1)/11 - 0.5/11
            ax = fig.add_subplot(3, len(pcds), i * len(pcds) + ij + 1, projection='3d')
            ax.view_init(elev, azim)
            if np.shape(pcd)[1] == 3:
                ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=size, cmap='cool', vmin=0, vmax=1)
            else:
                ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=size, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(titles[ij])
            ax.set_axis_off()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)
