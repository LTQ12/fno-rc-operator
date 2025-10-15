# 测试信号

import  numpy as np
from cft_1d import CFT

def replace_fft_with_cft(signal, t_coords, n_intervals=5, interp_order=7):
    # 预处理步骤：生成CFT所需参数
    n_samples = len(signal)

    # 1. 生成时域分段区间
    xd = np.linspace(t_coords[0], t_coords[-1], n_intervals + 1)

    # 2. 生成插值参数
    n1 = np.full(n_intervals, interp_order)
    MM = n1 * 2  # 示例值，实际需根据插值方法计算

    # 3. 计算拉格朗日插值系数
    D = [compute_lagrange_coeffs(interp_order) for _ in range(n_intervals)]

    # 4. 预计算阶乘矩阵
    MS = [precompute_factorial_matrix(interp_order) for _ in range(n_intervals)]

    # 5. 生成子区间边界和插值点
    x_B, C = generate_intervals_and_points(t_coords, n_intervals, interp_order)

    # 6. 生成频域采样点（需与FFT对齐）
    sampling_rate = 1 / (t_coords[1] - t_coords[0])
    f_points = np.fft.fftfreq(n_samples, d=1 / sampling_rate)

    # 调用CFT
    Ff, _ = CFT(
        xd=xd,
        f=signal,  # 假设信号已按区间预处理
        u=f_points,
        n1=n1,
        MM=MM,
        M=interp_order + 1,  # M为插值节点数+1
        D=D,
        MS=MS,
        x_B=x_B,
        C=C
    )

    # 结果对齐（可能需要相位调整）
    return np.fft.fftshift(Ff) * (t_coords[1] - t_coords[0])  # 根据CFT的积分系数调整


import numpy as np
from scipy.special import factorial
from numpy.polynomial.polynomial import polyfromroots


def compute_lagrange_coeffs(order: int) -> np.ndarray:
    """
    计算拉格朗日插值基函数系数矩阵

    参数:
    order - 插值阶数(M-1)

    返回:
    D - 系数矩阵，形状为(M, M)
    """
    # 生成等距插值节点（也可改为切比雪夫节点）
    nodes = np.linspace(-1, 1, order + 1)
    M = order + 1

    D = np.zeros((M, M))
    for i in range(M):
        # 获取第i个基函数的根
        roots = np.delete(nodes, i)

        # 构造多项式系数
        numerator = polyfromroots(roots)

        # 计算分母
        denominator = np.prod(nodes[i] - roots)

        # 归一化系数
        D[i, :] = numerator / denominator

    return D


def precompute_factorial_matrix(max_order: int) -> np.ndarray:
    """
    预计算阶乘矩阵

    参数:
    max_order - 最大阶数

    返回:
    MS - 阶乘矩阵，形状为(S+1, M)
    """
    M = max_order + 1
    MS = np.zeros((max_order + 1, M))

    for s in range(max_order + 1):
        for m in range(s, M):
            MS[s, m] = (-1) ** (m - s) / (factorial(m - s, exact=True))

    return MS


def generate_intervals_and_points(t_coords: np.ndarray,
                                  n_intervals: int,
                                  interp_order: int) -> tuple:
    """
    生成子区间边界和插值点

    参数:
    t_coords - 原始时间坐标
    n_intervals - 区间数量
    interp_order - 插值阶数

    返回:
    x_B - 区间边界列表 [[x0, x1], [x1, x2], ...]
    C - 每个区间的插值点坐标列表
    """
    # 划分区间边界
    xd = np.linspace(t_coords[0], t_coords[-1], n_intervals + 1)
    x_B = [[xd[i], xd[i + 1]] for i in range(n_intervals)]

    # 生成每个区间的插值点
    C = []
    for a, b in x_B:
        # 在[a, b]区间生成interp_order+1个等距点
        points = np.linspace(a, b, interp_order + 1)
        C.append(points.tolist())

    return x_B, C

t = np.linspace(0, 1, 1024)
signal = np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*120*t)

# 原FFT方法
fft_result = np.fft.fft(signal)

# 新CFT方法
cft_result = replace_fft_with_cft(signal, t)

# 比较主要频率成分
assert np.allclose(np.abs(fft_result), np.abs(cft_result), atol=1e-3)