.. _afpmsm:

=============================================
Axial-flux permanent magnet synchronous motor
=============================================

This electric motor model in FAST-OAD-CS23-HE represents a axial-flux PMSM that generates mechanical power through
interaction between the permanent magnets in the rotor and the electromagnets of winded wires. This component can be
activated through the powertrain configuration file (PT file). The registered installation positions can be found in
:ref:`options <options-afpmsm>`.

.. code-block:: yaml

    power_train_components:
      ⋮
      motor_1:
        id: fastga_he.pt_component.pmsm
        position: ...


A brief description of the SM PMSM component is presented here:

.. _table:
.. toctree::
   :maxdepth: 2

    AF PMSM computation logic <models>
    AF PMSM options <options>
    AF PMSM model assumptions <assumptions>