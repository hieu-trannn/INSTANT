# INSTANT: COMPRESSING GRADIENTS AND ACTIVATIONS FOR RESOURCE-EFFICIENT TRAINING
> Tuan-Kiet Doan, Trung-Hieu Tran, Enzo Tartaglione, Nikola Simidjievski & Van-Tam Nguyen

Official repository for the paper `INSTANT: COMPRESSING GRADIENTS AND ACTIVATIONS FOR RESOURCE-EFFICIENT TRAINING` accepted in [ICLR 2026](https://openreview.net/forum?id=P2q6Y7UweV).

<details><summary>Abstract</summary>
Deep learning has advanced at an unprecedented pace. This progress has led to a significant increase in its complexity. However, despite extensive research on accelerating inference, training deep models directly within a resource-constrained
budget remains a considerable challenge due to its high computational and memory requirements. In this paper, we introduce INSTANT (compressIng gradieNtS and acTivAtions for resource-efficieNt Training), a method designed to address both
the computational and the memory bottlenecks when training. INSTANT reduces resource demands during backpropagation by projecting gradients and activations into a low-rank subspace and performing computation within that compressed representation. Experimental results demonstrate that INSTANT achieves a 15× reduction in computational cost and 32× reduction in activation memory with negligible impact on model performance. 

</details>
## Compressing Gradient and Activation
<p align="center">
  <img src="figures/intro.pdf" width="300"/>
</p>

</details>
## Low-rank backpropagation
<p align="center">
  <img src="figures/low-rank_instant.pdf" width="300"/>
</p>

# Computer Vision tasks and Language tasks
Please follow directory `computer_vision` and `language`, repectively.

# References
If you use this code, plese consider citing this work as:
```
@inproceedings{
doantran2026instantcompressinggradientactivation,
title={INSTANT: COMPRESSING GRADIENTS AND ACTIVATIONS FOR RESOURCE-EFFICIENT TRAINING},
author={Tuan-Kiet Doan, Trung-Hieu Tran, Enzo Tartaglione, Nikola Simidjievski, and Van-Tam Nguyen},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=P2q6Y7UweV}
}
```

# License
See [LICENSE](LICENSE).

