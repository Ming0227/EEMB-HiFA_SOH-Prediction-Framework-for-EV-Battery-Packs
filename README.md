# EEMB-HiFA: SOH Prediction Framework for EV Battery Packs

This repository contains the code and materials for the study titled [**"Domain-adaptive State-of-Health Prediction of On-vehicle Batteries Powered by Deep Learning."**](https://doi.org/10.1016/j.xcrp.2025.102550) The proposed framework, EEMB-HiFA, integrates Two-Stage Mode Decomposition (TSMD) and a Hierarchically Fused Attention mechanism to achieve accurate and domain-adaptive State-of-Health (SOH) prediction for electric vehicle (EV) battery packs.

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

**Domain-adaptive state of health prediction of vehicle batteries powered by deep learning**  
Minghe Li, Zicheng Fei, Luoxiao Yang, Zijun Zhang, Kwok-Leung Tsui  
*Cell Reports Physical Science*, 2025, 102550, ISSN 2666-3864.  
[https://doi.org/10.1016/j.xcrp.2025.102550](https://doi.org/10.1016/j.xcrp.2025.102550)

BibTeX:
```bibtex
@article{li2025adaptive,
  title = {Domain-adaptive state of health prediction of vehicle batteries powered by deep learning},
  journal = {Cell Reports Physical Science},
  pages = {102550},
  year = {2025},
  issn = {2666-3864},
  doi = {https://doi.org/10.1016/j.xcrp.2025.102550},
  url = {https://www.sciencedirect.com/science/article/pii/S2666386425001493},
  author = {Minghe Li and Zicheng Fei and Luoxiao Yang and Zijun Zhang and Kwok-Leung Tsui}
}
```
---

## Contact

For questions, please contact:

Minghe Li: mingheli2-c@my.cityu.edu.hk
