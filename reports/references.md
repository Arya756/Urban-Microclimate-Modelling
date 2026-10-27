# Key references

The papers the Phase 1 report builds on, grouped by topic, with what this project takes from each. Full citations are in the reference list of [`Phase1_Report.pdf`](Phase1_Report.pdf). Most of these papers are published under their publishers' copyright, so they are linked rather than included here.

## Dataset and simulation

| Paper | What we use |
| --- | --- |
| Nazarian et al. (2025), UrbanTALES, *Bull. Amer. Meteor. Soc.* 106, E2461–E2478. [doi:10.1175/BAMS-D-25-0061.1](https://doi.org/10.1175/BAMS-D-25-0061.1) | The 538 LES cases, the metadata sheet and each case's friction velocity. The data are distributed by the authors under CC BY-NC 4.0. |
| Maronga et al. (2020), PALM model system 6.0, *Geosci. Model Dev.* 13, 1335–1372. [doi:10.5194/gmd-13-1335-2020](https://doi.org/10.5194/gmd-13-1335-2020) | The LES model that produced the cases. Its settings files give the forcing direction and the grid spacing used by the loader. |
| Deardorff (1980), *Boundary-Layer Meteorol.* 18, 495–527 | The 1.5-order sub-grid turbulence closure used in the simulations. |

## Urban canopy physics

| Paper | What we use |
| --- | --- |
| Grimmond and Oke (1999), *J. Appl. Meteorol.* 38, 1262–1292 | Morphometric description of a neighbourhood (plan-area density, frontal-area density, height statistics) and its link to the displacement height d and roughness length z0. |
| Macdonald et al. (1998), *Atmos. Environ.* 32, 1857–1864. [doi:10.1016/S1352-2310(97)00403-2](https://doi.org/10.1016/S1352-2310(97)00403-2) | The formulas for d and z0 of obstacle arrays, computed as features with α = 4.43, β = 1, C_D = 1.2 and κ = 0.4. |
| Kanda et al. (2013), *Boundary-Layer Meteorol.* 148, 357–377. [doi:10.1007/s10546-013-9818-x](https://doi.org/10.1007/s10546-013-9818-x) | Corrections of d and z0 for variable building heights, computed as features with the published coefficients. |
| Lindberg et al. (2018), UMEP, *Environ. Model. Softw.* 99, 70–87. [doi:10.1016/j.envsoft.2017.09.020](https://doi.org/10.1016/j.envsoft.2017.09.020) | Reference implementation against which the Kanda formulas were checked. |
| Bentham and Britter (2003), *Atmos. Environ.* 37, 2037–2043 | The canopy-averaged velocity U_c/u* = (2/λf)^(1/2), a physical reference for the role of frontal area. |
| Coceal and Belcher (2004), *Q. J. R. Meteorol. Soc.* 130, 1349–1372 | Canopy model of the mean wind profile, background for the canopy features. |
| Duan and Takemi (2021), *J. Appl. Meteorol. Climatol.* 60, 999–1018. [doi:10.1175/JAMC-D-20-0266.1](https://doi.org/10.1175/JAMC-D-20-0266.1) | Random forests predicting d and z0 from building morphology. |
| Lu et al. (2023), *J. Adv. Model. Earth Syst.* 15, e2022MS003287. [doi:10.1029/2022MS003287](https://doi.org/10.1029/2022MS003287) | Geometric parameters that separate realistic from idealized layouts, the reason both subsets are used for training. |

## Pedestrian-level wind and CFD

| Paper | What we use |
| --- | --- |
| Blocken and Stathopoulos (2013), *J. Wind Eng. Ind. Aerodyn.* 121, 138–145 | Why pedestrian-level wind matters: comfort, safety and ventilation. |
| Blocken (2015), *Build. Environ.* 91, 219–245. [doi:10.1016/j.buildenv.2015.02.015](https://doi.org/10.1016/j.buildenv.2015.02.015) | The cost and validation requirements of urban CFD, the motivation for a surrogate. |

## Learned flow surrogates

| Paper | What we use |
| --- | --- |
| Lu et al. (2025), *Environ. Res. Lett.* 20, 054006. [doi:10.1088/1748-9326/adc148](https://doi.org/10.1088/1748-9326/adc148) | The first surrogate trained on UrbanTALES: a U-Net and a random-forest baseline, evaluated on a random case split. The closest comparison for our results. |
| Snaiki et al. (2026), *Build. Environ.* 294, 114354. [doi:10.1016/j.buildenv.2026.114354](https://doi.org/10.1016/j.buildenv.2026.114354) | A U-Net refined by a conditional GAN, aimed at sharper peak speeds. |
| Huang et al. (2026), [arXiv:2607.02560](https://arxiv.org/abs/2607.02560) | An inpainting U-Net that stitches patch predictions across urban morphologies. |
| Clemente et al. (2023), [arXiv:2311.07985](https://arxiv.org/abs/2311.07985) | Configurable convolutional networks that trade accuracy against speed. |
| Guo et al. (2016), KDD, 481–490. [doi:10.1145/2939672.2939738](https://doi.org/10.1145/2939672.2939738) | Early convolutional approximation of steady flow. |
| Farimani et al. (2017), [arXiv:1709.02432](https://arxiv.org/abs/1709.02432) | Conditional GANs reproducing transport solutions. |
| Thuerey et al. (2020), *AIAA J.* 58, 25–36 | U-Nets for Reynolds-averaged airfoil flows. |

## Machine learning methods

| Paper | What we use |
| --- | --- |
| Breiman (2001), *Mach. Learn.* 45, 5–32 | Random forests. |
| Chen and Guestrin (2016), KDD, 785–794. [doi:10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785) | XGBoost. |
| Ke et al. (2017), *NeurIPS* 30 | LightGBM, the most accurate case-level model and the pixel model. |
| Rasmussen and Williams (2006), *Gaussian Processes for Machine Learning*, MIT Press | The Gaussian process with a Matérn kernel, one length scale per feature, fitted by marginal likelihood. |
| Pedregosa et al. (2011), *J. Mach. Learn. Res.* 12, 2825–2830 | scikit-learn, used for every model and split. |
| Akiba et al. (2019), KDD, 2623–2631; Bergstra et al. (2011), *NeurIPS* 24 | Optuna and the tree-structured Parzen estimator for the inner hyperparameter search. |
| Lundberg and Lee (2017), *NeurIPS* 30; Lundberg et al. (2020), *Nat. Mach. Intell.* 2, 56–67 | SHAP values and TreeSHAP for feature attribution. |
| Efron and Tibshirani (1993), *An Introduction to the Bootstrap*, Chapman & Hall | The paired percentile bootstrap over city and layout groups. |

## Evaluation with dependent data

| Paper | What we use |
| --- | --- |
| Roberts et al. (2017), *Ecography* 40, 913–929 | Grouped cross-validation for structured data. |
| Ploton et al. (2020), *Nat. Commun.* 11, 4540 | Evidence that spatial validation can sharply reduce apparent accuracy. |
| Varma and Simon (2006), *BMC Bioinformatics* 7, 91; Cawley and Talbot (2010), *J. Mach. Learn. Res.* 11, 2079–2107 | Selection bias from tuning on the scoring folds, the reason for nested cross-validation. |
| Kaufman et al. (2012), *ACM Trans. Knowl. Discov. Data* 6, 15 | Forms of leakage, the reason features never read simulated fields. |

## Planned for Phase 2

| Paper | What we use |
| --- | --- |
| Ronneberger et al. (2015), MICCAI, LNCS 9351, 234–241 | The U-Net architecture. |
| Perez et al. (2018), AAAI | FiLM conditioning of a network on the physics features. |
| Goodfellow et al. (2014), *NeurIPS* 27, 2672–2680 | Generative adversarial networks, used by refinement models. |
