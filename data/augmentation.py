import numpy as np
import paddle

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


def random_pcd_rotate(pcd: np.ndarray, max_rad: float = 0.785):
    """
    :param pcd: np.ndarray, shape=[n, 3]
    :param max_rad: float
    :return: np.ndarray, shape=[n, 3]
    """
    theta = np.random.uniform(-max_rad, max_rad)
    vec = np.random.rand(3)
    vec_unit = vec / np.linalg.norm(vec)
    r_vec = theta * vec_unit
    r_mat = geometry_utils.angleaxis_to_matrix(r_vec)
    pcd_rotated = np.matmul(r_mat, pcd)

    return pcd_rotated


def random_pcd_jitter(pcd: np.ndarray, alpha: float = 0.02, beta: float = 0.05):
    """
    :param pcd: np.ndarray, shape=[n, 3]
    :param alpha: float
    :param beta: float
    :return: np.ndarray, shape=[n, 3]
    """
    jitter = np.clip(alpha * np.random.randn(pcd.shape), -beta, beta)
    pcd_jittered = pcd + jitter

    return pcd_jittered


def normalize_pcd(pcd: np.ndarray):
    """
    :param pcd: np.ndarray, shape=[n, 3]
    :return: np.ndarray, shape=[n, 3]
    """
    mean = np.mean(pcd, axis=0)
    std = np.std(pcd, axis=0)
    pcd_normalized = (pcd - mean) / std

    return pcd_normalized


def augment_pcd(pcd, drop_rate=0.5, scale=1.25, shift=0.1, rotate_rad=0.785, jitter=(0.02, 0.05), use_normalize=True):
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
    if scale:
        pcd = random_pcd_scale(pcd, scale)
    if shift:
        pcd = random_pcd_shift(pcd, shift)
    if rotate_rad:
        pcd = random_pcd_rotate(pcd, rotate_rad)
    if jitter:
        pcd = random_pcd_jitter(pcd, jitter[0], jitter[1])

    return pcd


if __name__ == '__main__':
    a = np.random.randn(500, 3)
    print(a.shape)
    aa = random_pcd_dropout(a)
    print(aa.shape)
