# EEMB-HiFA: SOH Prediction Framework for EV Battery Packs

This repository contains the code and materials for the study titled **"Domain-adaptive State-of-Health Prediction of On-vehicle Batteries Powered by Deep Learning."** The proposed framework, EEMB-HiFA, integrates Two-Stage Mode Decomposition (TSMD) and a Hierarchically Fused Attention mechanism to achieve accurate and domain-adaptive State-of-Health (SOH) prediction for electric vehicle (EV) battery packs.

---

## Repository Contents

The repository includes the following key components:

- **`B_VMD.py`**:  
  Contains the implementation of the Variational Mode Decomposition for Batteries (B_VMD) algorithm, which is a critical component used in the TSMD module.

- **`TSMD.py`**:  
  Implements the Two-Stage Mode Decomposition (TSMD) method for SOH denoising and feature extraction from historical SOH data.

- **`HiFA.py`**:  
  Contains the code for the Hierarchically Fused Attention (HiFA) mechanism, which is integrated into the EEMB-HiFA framework for feature selection and fusion.

- **`tcn.py`**:  
  Provides the implementation of the Temporal Convolutional Network (TCN), which is used as a backbone for the proposed EEMB-HiFA model.

- **`EEMB-HiFA.py`**:  
  The main script that combines all components to implement the overall EEMB-HiFA architecture for SOH prediction.

---

## Citation

If you use this repository in your research, please cite the following paper:

@article{LI2025102550,
  title = {Domain-adaptive state of health prediction of vehicle batteries powered by deep learning},
  journal = {Cell Reports Physical Science},
  pages = {102550},
  year = {2025},
  issn = {2666-3864},
  doi = {https://doi.org/10.1016/j.xcrp.2025.102550},
  url = {https://www.sciencedirect.com/science/article/pii/S2666386425001493},
  author = {Minghe Li and Zicheng Fei and Luoxiao Yang and Zijun Zhang and Kwok-Leung Tsui},
  keywords = {lithium-ion battery, electric vehicles, state of health, deep learning, data-driven model, domain adaptation, signal denoising, data feature fusion},
  abstract = {State of health (SOH) estimation of battery packs in electric vehicles (EVs) is essential for transportation electrification safety and reliability. The noise and complexity of EV battery pack data hinder the effectiveness of various data-driven SOH estimation methods using lab data. To address these challenges and achieve more effective data-driven EV battery pack SOH predictions, this study develops a comprehensive deep-learning-based SOH modeling framework for EV batteries. The framework begins with a two-stage mode decomposition (TSMD) method designed to effectively identify neat SOH degradation patterns better representing noisy field data. Next, an endogenous and exogenous multibranch network structure with a hierarchically fused attention mechanism (EEMB-HiFA) is developed for real-time prediction of EV battery pack SOH. Computational experiments leveraging datasets from seven EVs are conducted to validate the accuracy and adaptiveness of the proposed EEMB-HiFA. The results show that the EEMB-HiFA can achieve a 96.49% improvement in accuracy compared to strong benchmarks considered.}
}

## Contact

For questions, please contact:

Minghe Li: mingheli2-c@my.cityu.edu.hk
