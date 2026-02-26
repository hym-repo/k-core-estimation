# Introduction

Estimate the maximum k-core number of uncorelated random networks.

# Files
- edgedata: edges of real-world networks
- real_theory:
    numerical iteration for solving equations and binary searching for kmax.
    The equation is proposed in "K-core organization of complex networks": 
    - $R = \sum_{n=0}^{k-2} \sum_{i=n}^{i_{\text{max}}-1} \frac{(i+1)P_{i+1}}{z_1} \cdot \binom{i}{n} \cdot R^{i-n}(1-R)^n$
- real_prun:
    pruning process: iteratively remove nodes with degree less than k
- real_estimate:
    the estimator proposed

    