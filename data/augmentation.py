import sys
import numpy as np
import paddle

sys.path.append('./')
from utils import geometry_utils


def random_pcd_dropout(pcd: np.ndarray, dropout_rate: float = 0.5):
    """
    :param pcd: np.ndarray, shape=[N, 3]
    :param dropout_rate: float
    :return: np.ndarray, shape=[n, 3]
    """
    drop_threshold = np.random.rand(1) * dropout_rate
    pcd_score = np.random.rand(pcd.shape[0])
    drop_index = np.where(pcd_score >= drop_threshold)
    pcd_sampled = pcd[drop_index]
    if pcd_sampled.shape[0] < 1:
        pcd_sampled = pcd

    return pcd_sampled


def random_pcd_scale(pcd: np.ndarray, max_scale: float = 1.25):
    """
    :param pcd: np.ndarray, shape=[n, 3]
    :param max_scale: float
    :return: np.ndarray, shape=[n, 3]
    """
    min_scale = 1. / max_scale
    scale = np.random.uniform(min_scale, max_scale)
    pcd_scaled = pcd * scale

    return pcd_scaled


def random_pcd_shift(pcd: np.ndarray, max_shift: float = 0.1):
    """
    :param pcd: np.ndarray, shape=[n, 3]
    :param max_shift: float
    :return: np.ndarray, shape=[n, 3]
    """
    shift = np.random.uniform(-max_shift, max_shift)
    pcd_shifted = pcd + shift

    return pcd_shifted


def random_pcd_rotate(pcd: np.ndarray, max_rad: float = 2 * np.pi):
    """
    :param pcd: np.ndarray, shape=[n, 3]
    :param max_rad: float
    :return: np.ndarray, shape=[n, 3]
    """
    # theta = np.random.uniform(-max_rad, max_rad)
    # vec = np.random.rand(3)
    # vec_unit = vec / np.linalg.norm(vec)
    # r_vec = theta * vec_unit
    # r_mat = geometry_utils.angleaxis_to_matrix(r_vec.reshape([3, -1]))
    # pcd_rotated = np.matmul(r_mat, pcd.T).T
    angle = np.random.uniform() * max_rad
    cosval = np.cos(angle)
    sinval = np.sin(angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1]], dtype=np.float32)
    pcd_rotated = np.matmul(rotation_matrix, pcd.T).T

    return pcd_rotated


def random_pcd_jitter(pcd: np.ndarray, alpha: float = 0.02, beta: float = 0.1):
    """
    :param pcd: np.ndarray, shape=[n, 3]
    :param alpha: float
    :param beta: float
    :return: np.ndarray, shape=[n, 3]
    """
    jitter = np.clip(alpha * np.random.randn(*pcd.shape), -beta, beta)
    pcd_jittered = pcd + jitter

    return pcd_jittered


def normalize_pcd(pcd: np.ndarray):
    """
    :param pcd: np.ndarray, shape=[n, 3]
    :return: np.ndarray, shape=[n, 3]
    """
    # mean = np.mean(pcd, axis=0)
    # std = np.std(pcd, axis=0)
    # pcd_normalized = (pcd - mean) / std
    centroid = np.mean(pcd, axis=0)
    pcd = pcd - centroid
    m = np.max(np.sqrt(np.sum(pcd ** 2, axis=1)))
    pcd_normalized = pcd / m

    return pcd_normalized


def augment_pcd(pcd, drop_rate=0.5, scale=1.25, shift=0.1, rotate_rad=0.785, jitter=(0.02, 0.1), use_normalize=True):
    """
    :param pcd: np.ndarry, shape=[n, 3]
    :param drop_rate: float or None
    :param scale: float or None
    :param shift: float or None
    :param rotate_rad: float or None
    :param jitter: float or None
    :param use_normalize: bool
    :return: np.ndarry, shape=[n, 3]
    """
    if use_normalize:
        pcd = normalize_pcd(pcd)
    else:
        pcd = pcd
    if drop_rate:
        pcd = random_pcd_dropout(pcd, drop_rate)
    if rotate_rad:
        pcd = random_pcd_rotate(pcd, rotate_rad)
    if jitter:
        pcd = random_pcd_jitter(pcd, jitter[0], jitter[1])
    if scale:
        pcd = random_pcd_scale(pcd, scale)
    if shift:
        pcd = random_pcd_shift(pcd, shift)

    return pcd


def farthest_point_sample(pcd, npoint):
    """
    Input:
        pcd: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = pcd.shape
    xyz = pcd[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones(shape=[N]) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    pcd_sampled = pcd[centroids.astype(np.int32)]

    return pcd_sampled


if __name__ == '__main__':
    a = np.random.randn(500, 3)
    aa = augment_pcd(a)
    print(aa.shape)
