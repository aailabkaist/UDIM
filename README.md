# UDIM (Unknown Domain Inconsistency Minimization for Domain Generalization, ICLR 2024)

Official PyTorch implementation of
**["Unknown Domain Inconsistency Minimization for Domain Generalization"](https://openreview.net/forum?id=eNoiRal5xi)** (ICLR 2024) 
by 
Seungjae Shin, 
HeeSun Bae, 
Byeonghu Na, 
Yoon-Yeong Kim, 
and Il-chul Moon.

## Overview

UDIM reduces the loss landscape inconsistency between source domain and unknown domains. 
As unknown domains are inaccessible, these domains are empirically crafted by perturbing instances from the source domain dataset. 
In particular, by aligning the loss landscape acquired in the source domain to the loss landscape of perturbed domains, 
we expect to achieve generalization grounded on these flat minima for the unknown domains.

![1](https://github.com/aailabkaist/UDIM/assets/20755743/b445eeba-02a9-40d4-9f9d-fc7241b3058f)

## Reproduce
For reproduce the results of LCMAT-S, we provide a bash file for running `main.py`, which located at: 
```
/bash/LCMat_XXX.sh
```
Here, XXX is dataset. You can get results in `result/` directory.
