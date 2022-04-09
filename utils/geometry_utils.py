import numpy as np
import pyquaternion


def vec_to_reversed_matrix(vec):
    """
    transfer 3-d vector to its reversed matrix, vec^
    :param vec: np.ndarray, shape=[3, 1]
    :return: mat: np.ndarray, shape=[3, 3]
    """
    a, b, c = vec[:, 0]
    mat = np.array([[0, -c, b],
                    [c, 0, -a],
                    [-b, a, 0]])

    return mat


def angleaxis_to_matrix(vec):
    """
    transfer so3 to SO3, use Rodriguez Formula, mat = cos(theta)I + (1-cos(theta))* nn^T + sin(theta)n^
    :param vec: np.ndarray, shape=[3, 1]
    :return: np.ndarray, shape=[3, 3]
    """
    theta = np.linalg.norm(vec)
    vec = vec / theta
    mat = np.cos(theta) * np.eye(3) + (1 - np.cos(theta)) * np.matmul(vec, vec.T) + np.sin(
        theta) * vec_to_reversed_matrix(vec)

    return mat


def matrix_to_angleaxis(mat):
    """
    transfer SO3 to so3, theta = arccos( (tr(R) - 1) / 2 ), reversed mat for vec = (R - R^T) / 2sin(theta)
    :param mat: np.ndarray, shape=[3, 3]
    :return: np.ndarray, shape=[3, 1]
    """
    theta = np.arccos((np.trace(mat) - 1.) / 2.)
    mat = ((mat - mat.T) / 2.) / np.sin(theta)
    a = -mat[1, 2]
    b = mat[0, 2]
    c = -mat[0, 1]
    vec = np.array([[a], [b], [c]]) * theta

    return vec


def euler_to_matrix(euler, angle_type='rad'):
    """
    transfer euler angle(RPY) roll, pitch, yaw to rotation matrix, note the spin axes are fixed
    equal to euler angle(ZYX), note the spin axes are unfixed
    R = R(z)R(y)R(x)
    :param euler: np.ndarray, shape=[3, 1], order=[gamma, beta, alpha]
    :param angle_type: str, 'degree' or 'rad'
    :return: np.ndarray, shape=[3, 3]
    """
    # gamma for X, beta for Y, alpha for Z
    assert angle_type in ['degree', 'rad'], 'Unsupported angle type {}'.format(angle_type)
    if angle_type == 'degree':
        euler = euler * np.pi / 180
    else:
        pass

    gamma = euler[0, 0]
    beta = euler[1, 0]
    alpha = euler[2, 0]
    rxg = np.array([[1, 0, 0],
                    [0, np.cos(gamma), -np.sin(gamma)],
                    [0, np.sin(gamma), np.cos(gamma)]])
    ryb = np.array([[np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)]])
    rza = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                    [np.sin(alpha), np.cos(alpha), 0],
                    [0, 0, 1]])
    mat = np.matmul(np.matmul(rza, ryb), rxg)

    return mat


def matrix_to_euler(matrix, angle_type='rad'):
    """
    transfer rotation matrix to euler angle(RPY) roll, yaw, pitch
    equal to euler angle(ZYX)
    :param matrix: np.ndarray, shape=[3, 3]
    :param angle_type: str, 'degree' or 'rad'
    :return: np.ndarray, shape=[3, 1]
    """
    assert angle_type in ['degree', 'rad'], 'Unsupported angle type {}'.format(angle_type)

    beta = np.arctan2(-matrix[2, 0], np.sqrt(np.square(matrix[0, 0]) + np.square(matrix[1, 0])))
    if beta != np.pi / 2 and beta != -np.pi / 2:
        alpha = np.arctan2(matrix[1, 0]/np.cos(beta), matrix[0, 0]/np.cos(beta))
        gamma = np.arctan2(matrix[2, 1]/np.cos(beta), matrix[2, 2]/np.cos(beta))
    elif beta == np.pi / 2:
        alpha = 0.
        gamma = np.arctan2(matrix[0, 1], matrix[1, 1])
    else:
        alpha = 0.
        gamma = -np.arctan2(matrix[0, 1], matrix[1, 1])

    if angle_type == 'degree':
        euler = np.array([[gamma], [beta], [alpha]]) * 180 / np.pi
    else:
        euler = np.array([[gamma], [beta], [alpha]])

    return euler


def quaternion_to_matrix(quaternion):
    """
    transfer quaternion (w, x, y, z) to rotation matrix
    :param quaternion: np.ndarray, shape=[4, 1]
    :return: np.ndarray, shape=[3, 3]
    """
    w = quaternion[0, 0]
    x = quaternion[1, 0]
    y = quaternion[2, 0]
    z = quaternion[3, 0]

    return np.array([[1. - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                     [2*x*y + 2*z*w, 1. - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
                     [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1. - 2*x*x - 2*y*y]])


def matrix_to_quaternion(matrix):
    """
    transfer rotation matrix to quaternion (w, x, y, z)
    :param matrix: np.ndarray, shape=[3, 3]
    :return: np.ndarray, shape=[4, 1]
    """
    w = np.sqrt(np.trace(matrix) + 1.) / 2.
    x = (matrix[2, 1] - matrix[1, 2]) / (4. * w)
    y = (matrix[0, 2] - matrix[2, 0]) / (4. * w)
    z = (matrix[1, 0] - matrix[0, 1]) / (4. * w)

    return np.array([[w], [x], [y], [z]])


def get_transformation_matrix(rotation, translation):
    """
    construct SE3 from rotation matrix and translation vector
    :param rotation: np.ndarray, shape=[3, 3]
    :param translation: np.ndarray, shape=[3, 1]
    :return: np.ndarray, shape=[4, 4]
    """
    rotation = np.insert(rotation, 3, 0, axis=0)
    translation = np.insert(translation, 3, 1, axis=0)
    t_mat = np.concatenate([rotation, translation], axis=-1)

    return t_mat


def get_relative_transformation_matrix(current, reference):
    """
    get relative transformation matrix, i.e. current pose in reference frame
    note that the notation of current pose is oTc, the notation of reference pose is oTd,
    then the relative dTc can be calculated as oTd^(-1)oTc
    :param current: np.ndarray, shape=[4, 4]
    :param reference: np.ndarray, shape=[4, 4]
    :return: np.ndarray, shape=[4, 4]
    """
    t_mat = np.matmul(np.linalg.inv(reference), current)

    return t_mat


def xyzabg_to_transformation_matrix(xyzabg):
    """
    transfer (x, y, z, alpha, beta, gamma) to transformation matrix
    note that alpha beta gamma for fixed ZYX axes, or unfixed XYZ axes
    :param xyzabg: np.ndarray, shape=[6, 1] or 6-d list
    :return: np.ndarray, shape=[4, 4]
    """
    xyzabg = np.array(xyzabg).reshape([6, 1])
    r_mat = euler_to_matrix(xyzabg[3:, :], angle_type='degree')
    t_vec = xyzabg[:3, :]
    t_mat = get_transformation_matrix(r_mat, t_vec)

    return t_mat


if __name__ == '__main__':
    vec = np.array([[-0.0042691901518842],
                    [0.008202525696573983],
                    [-0.01332348400744696]])
    mat = angleaxis_to_matrix(vec)
    print(mat)
    vec = matrix_to_angleaxis(mat)
    print(vec)
    rpy = matrix_to_euler(mat)
    print(rpy)
    quat = matrix_to_quaternion(mat)
    print(quat)
    mat = quaternion_to_matrix(quat)
    print(mat)
    mat = get_transformation_matrix(mat, vec)
    print(mat)


