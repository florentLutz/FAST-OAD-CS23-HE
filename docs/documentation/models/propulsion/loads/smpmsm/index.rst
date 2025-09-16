.. _smpmsm:

==================================================
Surface-mounted permanent magnet synchronous motor
==================================================

This electric load model in FAST-OAD-CS23-HE represents a surface mounted PMSM that generates mechanical power through
interaction between the permanent magnets at the rotor and the electromagnets of winded wires.



This component can be activated through the powertrain configuration file (PT file). The registered installation
positions and polarization model option can be found in the description of the PEMFC stack position and model fidelity option
in :ref:`options <options-pemfc>`.

.. code-block:: yaml

    power_train_components:
      ⋮
      pemfc_stack_1:
        id: fastga_he.pt_component.sm_pmsm
        position: ...


A brief description of the SM PMSM component is presented here:

.. _table:
.. toctree::
   :maxdepth: 2

    SM PMSM computation logic <models>
    SM PMSM options <options>
    SM PMSM model assumptions <assumptions>