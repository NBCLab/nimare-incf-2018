# nimare-incf-2018
Materials for the INCF Neuroinformatics 2018 poster, "NiMARE: Neuroimaging Meta-Analysis Research Environment".

## Authors
Taylor Salo, Katherine L. Bottenhorn, Thomas E. Nichols, Michael C. Riedel, Matthew T. Sutherland, Tal Yarkoni, and Angela R. Laird

## Abstract
We present NiMARE, a Python package for performing meta-analyses of the neuroimaging literature. While meta-analytic packages exist which implement one or two algorithms each, NiMARE provides a standard syntax for performing a wide range of analyses and for interacting with databases of coordinates and images from fMRI studies (e.g., brainspell, Neurosynth, and Neurovault). NiMARE joins a growing Python ecosystem for neuroimaging research, which includes such tools as nipype, nistats, and nilearn. As with these other tools, NiMARE is open source, collaboratively developed, and built with ease of use in mind.

As a demonstration of NiMARE’s capabilities, we have performed a series of image- and coordinate-based meta-analyses of a set of 21 fMRI studies of pain taken from Neurovault. NiMARE contains implementations of the nine image-based meta-analytic methods described in Maumet and Nichols (2016; MFX GLM, RFX GLM, FFX GLM, Contrast Permutation, Fisher’s, Stouffer’s, Weighted Stouffer’s, Z MFX, and Z Permutation), as well as five coordinate-based methods (ALE, SCALE, MKDA density analysis, MKDA chi-square analysis, and KDA density analysis). Results from each of these meta-analyses are shown in Figure 1. As would be expected, the image-based fixed effect procedures are much more sensitive than mixed and random effects, and the coordinate-based methods find less expansive regions of activation but generally pick up most of the pain network.

## Figures
### Figure 1. Meta-analytic results.
![alt text](https://github.com/NBCLab/nimare-incf-2018/blob/master/figures/metas.png "Meta-analytic results")

---

### Figure 2. Comparison of z-statistics.
![alt text](https://github.com/NBCLab/nimare-incf-2018/blob/master/figures/z_comparison.png "Comparison of z-statistics")

## References
- Maumet, C., & Nichols, T. E. (2016). Minimal Data Needed for Valid & Accurate Image-Based fMRI Meta-Analysis. *bioRxiv*, 048249. doi: [10.1101/048249](https://doi.org/10.1101/048249)
