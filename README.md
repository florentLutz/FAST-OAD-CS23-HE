![](FAST_OAD_logo.jpg) 

FAST-OAD-CS23-HE: An open-source framework for the design and analysis of hybrid electric General Aviation aircraft
======================================================================================
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Documentation Status](https://readthedocs.org/projects/fast-oad-cs23-he/badge/?version=stable)](https://fast-oad-cs23-he.readthedocs.io/)
![Tests](https://github.com/florentLutz/FAST-OAD-CS23-HE/workflows/Tests/badge.svg)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

FAST-OAD-CS23-HE is a plugin from the [FAST-OAD framework](https://github.com/fast-aircraft-design/FAST-OAD) performing rapid Overall Aircraft Design of hybrid electric aircraft. It inherits the aircraft sizing methods from [FAST-OAD-GA](https://github.com/supaero-aircraft-design/FAST-GA) and is complemented with a library of hybrid-electric powertrain components. The aircraft powertrain is described as a graph in what is called the powertrain file which is then used to automatically build te corresponding OpenMDAO problem.

This repository was created to contain the files created for the thesis *Optimization of an 
aircraft design problem for hybrid-electric configurations
under manufacturing and certification constraints* conducted by Florent Lutz and *Intégration de la discipline de gestion 
thermique dans la conception conceptuelle et préliminaire de nouvelles configurations d'avions* conducted by Valentine Habrard

Citation
--------

If you want to use FAST-OAD-CS23_HE for your work, or if you want more details about the organisation and functioning of the work, please consider citing the following papers:

```
@article{lutz2025open,
author = {Lutz, Florent and Jezegou, Joel and Budinger, Marc and Reysset, Aurelien},
title = {Open-Source Framework for Sizing Hybrid and Electric General Aviation Aircraft},
journal = {Journal of Aircraft},
volume = {62},
number = {2},
pages = {381-395},
year = {2025},
doi = {10.2514/1.C038004},
URL = {https://doi.org/10.2514/1.C038004},
eprint = {https://doi.org/10.2514/1.C038004},
abstract = { This paper presents FAST-OAD-CS23-HE, an open-source framework that has been developed as an extension of the existing FAST-OAD-GA framework to allow for medium fidelity sizing of hybrid and electric aircraft using component models dependent on operating conditions and sizing criteria. It inherits Overall Aircraft Design methodologies from the original framework and adds a library of models to represent physical components for hybrid powertrains. It also adds a generic methodology that enables the extension to multiphysic simulation for the powertrain and the consideration of synergistic interactions and supports the addition of new components. A graph-based description of the powertrain was chosen to easily describe complex powertrains that could be considered in innovative architectures as well as ease the future interfacing with external tools. In addition to that, the graph-based approach allows the automation of the construction of the design problem, which removes the need for users to handle complex scripts. It has been developed after a comparison of existing methodologies and open-source framework in an effort to bridge the gap in terms of preliminary design of electric aircraft. With the models implemented with the default delivery of the code, two aircraft were studied to serve as a reference to showcase the capabilities of this framework. }
}

@article{habrard2025parametric,
author = {Habrard, Valentine and Pommier-Budinger, Valérie and Hazyuk, Ion and Jézégou, Joël and Benard, Emmanuel},
title = {Parametric Study of a Liquid Cooling Thermal Management System for Hybrid Fuel Cell Aircraft},
journal = {Aerospace},
volume = {12},
year = {2025},
number = {5},
article-number = {377},
url = {https://www.mdpi.com/2226-4310/12/5/377},
issn = {2226-4310},
abstract = {Hybrid aircraft offer a logical pathway to reducing aviation’s carbon footprint. The thermal management system (TMS) is often neglected in the assessment of hybrid aircraft performance despite it being of major importance. After presenting the TMS architecture, this study performs a sensitivity analysis on several parameters of a retrofitted hybrid fuel cell aircraft’s performance considering three hierarchical levels: the aircraft, fuel cell system, and TMS component levels. The objective is to minimize CO2 emissions while maintaining performance standards. At the aircraft level, cruise speed, fuel cell power, and ISA temperature were varied to assess their impact. Lowering cruise speeds can decrease emissions by up to 49%, and increasing fuel cell power from 200 kW to 400 kW cuts emissions by 18%. Higher ambient air temperatures also significantly impact cooling demands. As for the fuel cell, lowering the stack temperature from 80 °C to 60 °C increases the required cooling air mass flow by 49% and TMS drag by 40%. At the TMS component level, different coolants and HEX offset-fin geometries reveal low-to-moderate effects on emissions and payload. Overall, despite some design choice improvements, the conventional aircraft is still able to achieve lower CO2 emissions per unit payload.},
doi = {10.3390/aerospace12050377}
}
```

Install
-------

To install the code without the optional dependencies linked with the LCA package, simply do the following:

* Clone the repository locally
* Preferably and although it is not mandatory, create a new virtual environment ([conda](https://docs.conda.io/en/latest/) is preferred).
* Install [Poetry](https://python-poetry.org/docs/) with a version greater than 1.8.3. We recommend installing poetry using the pipx method and making use of suffixes.
* At root of the project folder, do `poetry install`

If you want to install the optional dependencies replace the last step with: 

* At root of the project folder, do `poetry install --extras lca`

For the LCA package to run, an ecoinvent license is required. For more details on the functioning of the LCA module, check out the [official documentation](https://fast-oad-cs23-he.readthedocs.io/en/latest/documentation/models/lca/index.html)