# DRMRR: Distributionally Robust Learning-To-Rank Under the Wasserstein Metric


Python implementation of models introduced in the paper "Distributionally robust learning-to-rank under the Wasserstein metric" by Shahabeddin Sotudian et. al.
Regardless of their satisfactory performance, most existing listwise {\em Learning-To-Rank (LTR)} models overlooked the crucial matter of robustness. A data set can be contaminated in various ways, including human error in labeling or annotation, distributional data shift, and malicious adversaries who wish to degrade the algorithm’s performance. It has been shown that {\em Distributionally Robust Optimization (DRO)} is resilient against various types of noise and perturbations. To fill this gap, we introduce a new listwise LTR model called {\em Distributionally Robust Multi-output Regression Ranking (DRMRR)}. Different from existing methods, the scoring function of DRMRR was designed as a multivariate mapping from a feature vector to a vector of deviation scores, which captures local context information and cross-document interactions. In this way, we are able to incorporate the LTR metrics into our model. DRMRR uses a Wasserstein DRO framework to minimize a multi-output loss function under the most adverse distributions in the neighborhood of the empirical data distribution defined by a Wasserstein ball. We present a compact and computationally solvable reformulation of the min-max formulation of DRMRR. Our experiments were conducted on two real-world applications: medical document retrieval and drug response prediction, showing that DRMRR notably outperforms state-of-the-art LTR models. We also conducted an extensive analysis to examine the resilience of DRMRR against various types of noise: Gaussian noise, adversarial perturbations, and label poisoning. Accordingly, DRMRR is not only able to achieve significantly better performance than other baselines, but it can maintain a relatively stable performance as more noise is added to the data.
 
 
## Citation

If you use the code, please cite this paper:

```text
@article{sotudianDRMRR,
  title={Distributionally robust learning-to-rank under the Wasserstein metric},
  author={Shahabeddin Sotudian, Ruidi Chen, and Ioannis Ch. Paschalidis},
  journal={},
  volume={},
  number={},
  pages={},
  year={},
  publisher={}
}
```
