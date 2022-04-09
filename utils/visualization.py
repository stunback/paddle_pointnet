import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d


def vis_pcd(pcd: np.ndarray):
    """
    visualize point cloud using plt
    :param pcd: np.ndarray, shape=[n, 3]
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], s=5, marker='o', color='b')
    plt.show()
