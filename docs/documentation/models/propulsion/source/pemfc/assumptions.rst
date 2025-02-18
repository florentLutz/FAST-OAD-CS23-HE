.. _assumptions-pemfc:

==============================================
Proton-exchange membrane fuel cell assumptions
==============================================
The followings are the assumptions that applied in the Proton-Exchange Membrane Fuel Cell (PEMFC) computation.

* All the gas are consider as ideal gas.
* The operating temperature of PEMFC remains equivalent to the ambient temperature.
* The hydrogen operating pressure of the analytical polarization model remains constant as the user defined in source file.
* The oxygen partial pressure of the analytical polarization model is set to 21% of the operating pressure.
* The operating pressure remains equivalent to the ambient pressure if no compressor is connected.
* The pressure of both electrodes in simple polarization model is assumed equivalent.
* The maximum current density is set to 0.7 [A/cm^2] obtained from :cite:`hoogendoorn:2018`.
