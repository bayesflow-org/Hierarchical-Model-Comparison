# Hierarchical Model Comparison


<img src="img/hierarchical_BMC.png" width=80% height=80%>

This repository contains the code for running the experiments and reproducing all results reported in our paper [A Deep Learning Method for Comparing Bayesian Hierarchical Models](https://arxiv.org/abs/2302.09125). We propose a deep learning method for performing Bayesian model comparison on any set of hierarchical models which can be instantiated as probabilistic programs. The method formulates the problem as data compression (i.e., embedding hierarchical data sets into informative summary vectors) and probabilistic classification (i.e., assigning posterior probabilities to the summary vectors). 

The details of the method are described in our paper:

Elsemüller, L., Schnuerch, M., Bürkner, P. C., & Radev, S. T. (2023). 
A Deep Learning Method for Comparing Bayesian Hierarchical Models
<em>arXiv preprint arXiv:2301.11873</em>, available for free at: https://arxiv.org/abs/2301.11873.

The code depends on the [BayesFlow](https://github.com/stefanradev93/BayesFlow) library, which implements the neural network architectures and training utilities.

## Cite

```bibtex
@article{elsemuller2023deep,
  title={A deep learning method for comparing bayesian hierarchical models},
  author={Elsem{\"u}ller, Lasse and Schnuerch, Martin and B{\"u}rkner, Paul-Christian and Radev, Stefan T},
  journal={arXiv preprint arXiv:2301.11873},
  year={2023}
}
```

## [notebooks](notebooks)

The experiments are structured as self-contained Jupyter notebooks, which are detailed below.

### [01_calibration_validation](notebooks/01_calibration_validation)

Code for reproducing the calibration experiments of validation study 1 that are composed of three sub-parts:
- [01_calibration_fixed_sizes](notebooks/01_calibration_validation/01_calibration_fixed_sizes.ipynb): Training and calibration assessment with data sets that all possess the same amount of groups and nested observations.
- [02_calibration_variable_observations](notebooks/01_calibration_validation/02_calibration_variable_observations.ipynb): Training and calibration assessment with data sets that all possess the same amount of groups but vary in their amount of nested observations.
- [03_calibration_variable_sizes](notebooks/01_calibration_validation/03_calibration_variable_sizes.ipynb): Training and calibration assessment with data sets that vary in their amount of groups as well as nested observations.

### [02_bridge_sampling_comparison](notebooks/02_bridge_sampling_comparison)

- [02_bridge_sampling_comparison/nested_models](notebooks/02_bridge_sampling_comparison/nested_models): Code for reproducing the bridge sampling benchmarking of validation study 1, in which the approximation performance of the neural network is tested against bridge sampling on a toy example.
- [02_bridge_sampling_comparison/non_nested_models](notebooks/02_bridge_sampling_comparison/non_nested_models): Code for reproducing the calibration experiment and bridge sampling benchmarking of validation study 2, based on the comparison of hierarchical SDT and MPT models.

### [03_levy_flight_application](notebooks/03_levy_flight_application)

Code for reproducing the application study in which two variants of the drift diffusion model are compared to two variants of a Lévy flight model. Consists of five steps:
- [01_simulator](notebooks/03_levy_flight_application/01_simulator.ipynb): Simulate training and validation data.
- [02_pretrain_networks](notebooks/03_levy_flight_application/02_pretrain_networks.ipynb): Pretrain the network on simulated data with a reduced amount of trials per participant.
- [03_finetune_networks](notebooks/03_levy_flight_application/03_finetune_networks.ipynb): Fine-tune the network on simulated data that contains the same amount of trials per participant as the empirical data.
- [04_validate_networks](notebooks/03_levy_flight_application/04_validate_networks.ipynb): Validate the trained networks on new simulated data sets.
- [05_apply_networks](notebooks/03_levy_flight_application/05_apply_networks.ipynb): Apply the trained networks to the empirical data set.

Here, we re-analyzed data from [Jumping to Conclusion? A Lévy Flight Model of Decision Making](https://www.tqmp.org/RegularArticles/vol16-2/p120) by Eva Marie Wieschen, Andreas Voss, and Stefan T. Radev. The data set can be requested from the authors of the original study.

## [src](src)

Contains custom [Julia](src/julia) and [Python](src/python) functions that enable the analyses, including our proposed hierarchical neural network architecture.

## Support

This work supported by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy -– EXC-2181 - 390900948 (the Heidelberg Cluster of Excellence STRUCTURES) and -- EXC-2075 - 390740016 (the Stuttgart Cluster of Excellence SimTech), the Cyber Valley Research Fund (grant number: CyVy-RF-2021-16), and by the research training group "Statistical Modeling in Psychology" (SMiP, also supported by the DFG, German Research Foundation; GRK 2277). 

## License

MIT
