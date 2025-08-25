.. _options-gaseous-hydrogen-tank:

===========================
Gaseous hydrogen tank model
===========================

********************
Tank position option
********************
The gaseous hydrogen tank model has four possible installation position:

| ``wing_pod`` : Tank installed under the wing with user specified position.
| ``underbelly`` : Tank installed under and outside the fuselage.
| ``in_the_cabin`` : Tank installed at the middle section inside the fuselage.
| ``in_the_back`` : Tank installed at the rear section inside the fuselage.

********************
Multiple tank option
********************
This option is activated by changing the "number_of_tank" variable in the input file. Please note it will only be used for in-fuselage installation, as indicated in
the :ref:`assumptions <assumptions-gaseous-hydrogen-tank>`.

===  ================================================
n    Tank outer diameter compared to single tank case
===  ================================================
1    1
2    :math:`\frac{1}{2}`
3    :math:`\frac{1}{1 + \frac{2}{3}\sqrt{3}}`
4    :math:`\frac{1}{1 + \sqrt{2}}`
5    :math:`\frac{1}{1 + \sqrt{2(1+1/\sqrt{5})}}`
6    :math:`\frac{1}{3}`
7    :math:`\frac{1}{3}`
8    :math:`\frac{1}{1 + \csc(\pi/7)}`
9    :math:`\frac{1}{1 + \sqrt{2(2+\sqrt{2})}}`
===  ================================================

This table indicates the maximum diameter of tanks located inside the fuselage as given by :cite:`kravitz:1967`.
