![](FAST_OAD_logo.jpg) 

FAST-OAD-CS23-HE: An open-source framework for the design and analysis of hybrid electric General Aviation aircraft
======================================================================================

FAST-OAD-CS23-HE is a plugin from the [FAST-OAD framework](https://github.com/fast-aircraft-design/FAST-OAD) performing rapid Overall Aircraft Design of hybrid electric aircraft. It inherits the aircraft sizing methods from [FAST-OAD-GA](https://github.com/supaero-aircraft-design/FAST-GA) and is complemented with a library of hybrid-electric powertrain components. The aircraft powertrain is described as a graph in what is called the powertrain file which is then used to automatically build te corresponding OpenMDAO problem.

This repository was created to contain the files created for the thesis *Optimization of an 
aircraft design problem for hybrid-electric configurations
under manufacturing and certification constraints* and *Intégration de la discipline de gestion 
thermique dans la conception conceptuelle et préliminaire de nouvelles configurations d'avions*

Install
-------

* Clone the repository locally
* Preferably and although it is not mandatory, create a new virtual environment ([conda](https://docs.conda.io/en/latest/) is preferred).
* Install [Poetry](https://python-poetry.org/docs/) with a version greater than 1.4.2. We recommend installing poetry using the pipx method and making use of suffixes.
* At root of the project folder, do `poetry install`
