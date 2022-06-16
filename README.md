#  Charge mosaics on contact-electrified dielectrics result from polarity-inverting discharges

This is a repository of code accompanying the research article 

Yaroslav I. Sobolev, Witold Adamkiewicz, Marta Siek, Bartosz A. Grzybowski,
"Charge mosaics on contact-electrified dielectrics result from polarity-inverting discharges", *Nature Physics* (2022).

<!---
Bibtech code for citing the paper:

```
@article{2022chargepatterns,
  title={Charge mosaics on contact-electrified dielectrics result from polarity-inverting discharges},
  author={Sobolev, Yaroslav I and Adamkiewicz, Witold and Siek, Marta and Grzybowski, Bartosz A},
  journal  = "Nature Physics",
  volume   =  XX,
  number   =  XX,
  pages    = XX,
  month    =  XX,
  year     =  2022
}
```
-->

Experimental data and results of numerical calculations are too large (~16 Gb) for Github and 
are available from [separate repository in Harvard Dataverse](https://doi.org/10.7910/DVN/ZOFDKM):
[![](<https://img.shields.io/badge/Dataverse DOI-10.7910/DVN/ZOFDKM-orange>)](https://www.doi.org/10.7910/DVN/ZOFDKM)

## Installation

This code is compatible with Python 3.7

The primary Python dependencies are:

* [Matplotlib](https://matplotlib.org/) version 3.1.0
* [Matplotlib-scalebar](https://pypi.org/project/matplotlib-scalebar/) version 0.6.0
* [Numpy](https://numpy.org/) version 1.21.6
* [Scipy](https://scipy.org/) version 1.5.2

Other Python dependencies are standard -- come pre-installed with Anaconda distribution.

Running custom code controlling the experimental setups (Scanning Kelvin probe and setup for controlled delamination,
see `custom_code_for_experimental_setups` folder)
additionally requires:
* **32-bit** Python 3 
* [PyAutoGui](https://pyautogui.readthedocs.io/en/latest/) version 0.9.47
* [thorlabs_apt](https://pypi.org/project/thorlabs-apt/) module for Python
* [Thorlabs APT](https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=9019) drivers (**32-bit**).

Running the numerical model (see `macro-scale-model` folder) requires:
* [PyAutoGui](https://pyautogui.readthedocs.io/en/latest/) version 0.9.47
* [COMSOL Multiphysics](https://www.comsol.com/) version 5.4 or later, with AC/DC package

## Reproducing methods and figures from the paper

For constructing Paschen's law interpolator from literature data 
(Supplementary Figure S26-S27)
see `paschen_curve_approximating/paschen_curve_processing.py`
and `paschen_curve_approximating/paschen_curve_processing_for_argon.py`

For reproducing Scanning Kelvin Probe (SKP) (Figure 2c,f, Figure 3b, 
Supplementary Figure 5) see 
folder `data_processing/kelvinprobe_viewer`

For plotting sections of SKP maps see `data_processing/kelvinprobe_sections` folder

For reproducing Supplementary Figures S14-S18 
see `data_processing/kelvinprobe_viewer/moving_averages.py`

For processing electrometer data (Figure 4d, Figure 3b, Supplementary Figures S10d, S11d, S12c, S6d) see 
scripts named `data_processing/process_electrometer*.py`

For evaluating net charge from electrometer data (Figure 3b, Supplementary Figure S6d) 
see `data_processing/net_charge_extraction.py`

For processing direct optical detection os sparks see (Figure 4b,c), see 
scripts `data_processing/sparks_camera_processing*.py `

For detecting motion of delamination front (Supplementary Figures S10b, S11b, S12c) see scripts
`data_processing/delamination_front_finder*.py` and `data_processing/delamination_front_plotter*.py`

For XPS data plotting (Figure S8, S9) see `data_processing/plot-all-XPS.py`

For hygrometer calibration (Supplementary Figure S33)
see `data_processing/hygrometer_calibration`

For evaluation of adhesion work (Figure 3c, Supplementary Figures S6c,e)
see `data_processing/adhesion_work_extraction.py`