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
<p align="center">
  <img src="https://github.com/aailabkaist/UDIM/assets/20755743/b445eeba-02a9-40d4-9f9d-fc7241b3058f" alt="Image Description">
</p>

## Reproduce
To reproduce the results of UDIM, we provide a bash file, which located at: 
```
/bash/udim_pacs_xxx.sh
```
Here, XXX is a specific experimental setting. 
