#  Charge mosaics on contact-electrified dielectrics result from polarity-inverting discharges

This is a repository of code accompanying the research article 

Yaroslav I. Sobolev, Witold Adamkiewicz, Marta Siek, Bartosz A. Grzybowski,
"Charge mosaics on contact-electrified dielectrics result
from polarity-inverting discharges", *Nature Physics* (2022). DOI

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

## Installation

This code is compatible with Python 3.7

The primary dependencies are:

* [Matplotlib](https://matplotlib.org/) version 3.1.1
* [Numpy](https://numpy.org/) version 1.17.3

Other dependencies are standard -- come pre-installed with Anaconda.

For running the numerical model (see `macro-scale-model` folder) you would additionally need
COMSOL Multiphysics 5.4 or later with AC/DC package.

## Reproducing methods and figures from the paper
Experimental data is and results of numerical calculations are too large for Github and 
are available from separate repository in Harvard Dataverse: URL, DOI.

For reproducing Scanning Kelvin Probe (SKP) (Figure 2c,f, Supplementary Figure 5) see 
folder `data_processing/kelvinprobe_viewer`

For plotting sections of SKP maps see `data_processing/kelvinprobe_sections` folder

For processing electrometer data (Figure 4d, Supplementary Figures S10d, S11d, S12c) see 
scripts named `data_processing/process_electrometer*.py`

For processing direct optical detection os sparks see (Figure 4b,c), see 
scripts `data_processing/sparks_camera_processing*.py `

For detecting motion of delamination front (Supplementary Figures S10b, S11b, S12c) see scripts
`data_processing/delamination_front_finder*.py` and `data_processing/delamination_front_plotter*.py`

For XPS data plotting (Figure S8, S9) see `data_processing/plot-all-XPS.py`

