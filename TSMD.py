import pandas as pd
import torch
import os
from PyEMD import EMD
import warnings
from scipy.stats import entropy
from B_VMD import VMD
import random
import matplotlib.ticker as ticker
import numpy as np

# JSD divergence

def js_divergence(p, q):
    p = p / np.sum(p)
    q = q / np.sum(q)

    m = 0.5 * (p + q)

    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))

    jsd = 0.5 * (kl_pm + kl_qm)

    return jsd


def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    kl_divergence_p = entropy(p, m)
    kl_divergence_q = entropy(q, m)
    jsd = 0.5 * (kl_divergence_p + kl_divergence_q)
    return jsd


def divergence_evaluate(S, md, window_size, overlap):
    jsd_values = []

    for i in range(0, len(S) - window_size + 1, overlap):
        S_window = S[i:i + window_size]
        md_window = md[i:i + window_size]
        jsd_value = jensen_shannon_divergence(S_window, md_window)
        jsd_values.append(jsd_value)
    return np.mean(jsd_values)


# KL-divergence
def kl_divergence(p, q):
    p = np.array(p)
    q = np.array(q)
    kl_div = np.sum(p * np.log(p / q))
    return kl_div


# SSIM
def SSIM(signal1, signal2):
    mean_signal1 = np.mean(signal1)
    mean_signal2 = np.mean(signal2)

    var_signal1 = np.var(signal1)
    var_signal2 = np.var(signal2)

    cov_signal1_signal2 = np.cov(signal1, signal2)[0][1]

    k1 = 0.01
    k2 = 0.03
    L = np.max(signal1)

    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2

    ssim = ((2 * mean_signal1 * mean_signal2 + c1) * (2 * cov_signal1_signal2 + c2)) / ((mean_signal1 ** 2 + mean_signal2 ** 2 + c1) * (var_signal1 + var_signal2 + c2))

    return ssim


# SNR signal-to-noise ratio
def calculate_snr(A, B):
    # 计算能量B
    A = A
    B = B
    energy_B = np.sum(B ** 2)
    var_B = np.var(B)

    diff = A - B

    energy_A_B = np.sum(diff ** 2)
    energy_A_B_mean = np.mean((diff)**2)
    
    snr = 10 * np.log10(energy_B / energy_A_B)
    information_percentage = (1 - (energy_A_B_mean / var_B) * 100)
    return information_percentage  # snr


# variance_explained_ratio
def variance_explained_ratio(imfs, data):
    data_var = np.var(data)
    imfs_var = np.sum(np.var(imfs))
    return imfs_var / data_var


# Energy Preservation Ratio
def energy_preservation_ratio(imfs, data):
    data_energy = np.sum(data**2)
    imfs_energy = np.sum(imfs**2)
    return imfs_energy / data_energy


def check_monotonic_decreasing(data, threshold):
    if isinstance(data, np.ndarray):
        diff_count = np.sum(np.diff(data) <= 0)
        if diff_count / len(data) >= threshold:
            return True
        else:
            return False
    elif isinstance(data, pd.Series):
        diff_count = np.sum(np.diff(data.values) <= 0)
        if diff_count / len(data) >= threshold:
            return True
        else:
            return False
    elif isinstance(data, torch.Tensor):
        diff_count = torch.sum(data[1:] <= data[:-1])
        if diff_count.item() / len(data) >= threshold:
            return True
        else:
            return False
    else:
        raise ValueError("Data must be a numpy array, pandas Series, or torch Tensor")


def B_EMD(data, original_data, threshold):
    emd = EMD()

    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        s = data.squeeze().values
    elif isinstance(data, torch.Tensor):
        s = data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        s = data
    else:
        raise ValueError("Data must be a pandas DataFrame/Series, torch Tensor, or np.ndarray")
    i = 0
    while True:
        imfs = emd(s, max_imf=i)
        check_result = imfs[-1]
        if check_monotonic_decreasing(check_result, threshold):
            jsd = js_divergence(original_data, np.array(check_result))
            snr = calculate_snr(np.array(check_result), original_data)
            ver = variance_explained_ratio(np.array(check_result), original_data)
            ssim = SSIM(np.array(check_result), original_data)
            return check_result, jsd, snr, ver, ssim
        else:
            i += 1


def TSMD(original_data, threshold):
    u, _, _ = VMD(original_data, alpha=2000, tau=0.0, K=6, DC=0, init=1, tol=1e-6, threshold=0.7)
    signal = u[0]

    bvmd_result, jsd_bvmd, snr_bvmd, ver_bvmd, epr_bvmd = B_EMD(signal, original_data, threshold=0.50)

    return bvmd_result, jsd_bvmd, snr_bvmd, ver_bvmd, epr_bvmd


