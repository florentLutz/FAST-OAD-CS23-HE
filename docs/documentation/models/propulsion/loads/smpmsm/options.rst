.. _options-smpmsm:

=========================================================
Surface mounted permanent magnet synchronous motor option
=========================================================

The SMPMSM model has two possible installation positions shown as:

| ``on_the_wing`` : Servo motor mounted on the wing of aircraft
| ``in_the_nose`` : Servo motor installed inside the nose of aircraft


.. code-block:: yaml

    power_train_components:
      ⋮
      motor_1:
        id: fastga_he.pt_component.sm_pmsm
        position: ...


