=============================
Life Cycle Assessment options
=============================

The Life Cycle Assessment (LCA) module can be parametrized according to several criterion which are implemented under the form of group options. A description of those options is available here. There is one mandatory option, the rest are configured with default values which will be detailed here.

.. contents::

.. _mandatory-options-lca:

*****************
Mandatory options
*****************

Powertrain file path
====================

The LCA module computes the impact of the manufacturing, distribution and use phase of the aircraft which includes the production of the components of the hybrid electric powertrain as well as the emissions associated with their use (in flight emissions but also emissions linked with fuel or electricity production). As such it is necessary to know the architecture of the powertrain which is read from the powertrain file. The path to that file is given in the :code:`power_train_file_path` option.

.. code:: yaml

    model:
        lca:
            id: fastga_he.lca.legacy
            power_train_file_path: hybrid_propulsion.yml

.. _supplementary-options-lca:

*********************
Supplementary options
*********************

Component level breakdown
=========================

By default, the LCA module only breaks down the impact of the aircraft down to the phase level (manufacturing, distribution, use, ...). The :code:`component_level_breakdown` option allows to further break down the impact calculation to the contribution of each component of the power train to each phase. By default this option is set to :code:`False`.

.. code:: yaml

    model:
        lca:
            id: fastga_he.lca.legacy
            power_train_file_path: hybrid_propulsion.yml
            component_level_breakdown: True

.. note::

    As enabling this option makes the module the contribution of each component to each phase it adds a significant number of outputs to the OpenMDAO problem which may affect the computation time.

.. _impact-assessment-method-lca:

Impact assessment method
========================

Multiple impact assessment method are implemented in the LCA module to compute the environmental impact of a hybrid electric aircraft. Which one to use can be chosen by setting the :code:`impact_assessment_method` option. The following impact assessment method are available and the value at which to set the option for each of them is also given:

* For `Environmental Footprint version 3.1 <https://eplca.jrc.ec.europa.eu/LCDN/developerEF.html>`_ without long term effects use :code:`EF v3.1 no LT`
* For `Environmental Footprint version 3.1 <https://eplca.jrc.ec.europa.eu/LCDN/developerEF.html>`_ with long term effects use :code:`EF v3.1`
* For `Impact World+ v2.0.1 <https://www.impactworldplus.org/version-2-0-1/>`_ use :code:`IMPACT World+ v2.0.1`
* For `ReCiPe 2016 <https://pre-sustainability.com/articles/recipe/>`_ use :code:`ReCiPe 2016 v1.03`

.. code:: yaml

    model:
        lca:
            id: fastga_he.lca.legacy
            power_train_file_path: hybrid_propulsion.yml
            impact_assessment_method: EF v3.1

.. note::

    The version of Impact World+ implemented in the Ecoinvent database, :ref:`which is used in this module <models-lca>`, does not handle :ref:`normalization <normalization-options-lca>`.

EcoInvent version
=================

The LCA modules relies on the EcoInvent database to perform part of the Life Cycle Inventory phase and the Life Cycle Impact Assessment phase. It is possible to choose which version of the database to use by setting the :code:`ecoinvent_version` option.

.. code:: yaml

    model:
        lca:
            id: fastga_he.lca.legacy
            power_train_file_path: hybrid_propulsion.yml
            ecoinvent_version: 3.9.1

.. note::

    In the current version of the code, only the 3.9.1 version has been tested. It has thus been decided to only enable that version.

.. _airframe-material-lca:

Airframe material
=================

The first step of the LCA module is to write the :ref:`LCA configuration file <models-lca>` which is then provided to the :code:`lcav` package which turns it, with the help of the :code:`lca-algebraic` package, in symbolic expressions for the impacts (for more information see section :ref:`models-lca`). Consequently choices, like that of the airframe material, which affect the writing of the LCA configuration file must be declared as options as opposed to what is done in other modules (where they are defined as inputs). The materials in which the airframe is built can be declared using the :code:`airframe_material` option. In the current version of the code either aluminium or composite can be selected. By default the former is used.

.. code:: yaml

    model:
        lca:
            id: fastga_he.lca.legacy
            power_train_file_path: hybrid_propulsion.yml
            airframe_material: aluminium

.. note::
    As explained in the :ref:`assumptions-lca` section, we will consider that the landing gear is always made of steel.

Aircraft delivery method
========================

As for the option on the choice of the :ref:`airframe material <airframe-material-lca>`, the choice of the delivery method of the aircraft from manufacturer to user can be changed via the option :code:`delivery_method`. Two delivery methods are currently considered: either the aircraft is flown to the user (in which case the value :code:`flight` must be set) or it can be transported via train (which corresponds to the :code:`train` value). By default the former is used.

.. code:: yaml

    model:
        lca:
            id: fastga_he.lca.legacy
            power_train_file_path: hybrid_propulsion.yml
            delivery_method: flight

Electric mix considered
=======================

By default the LCA modules considers the average European electric mix for the manufacturing phase and for the charging of the batteries if there are any (See section :ref:`assumptions-lca` for more information). This default choice (:code:`default`) can be overridden for all higher level processes, meaning all process which appear explicitly in the :ref:`LCA configuration file <models-lca>`. The currently implemented alternatives include the French electric mix (:code:`french`) or the Slovenian electric mix (:code:`slovenia`).

.. code:: yaml

    model:
        lca:
            id: fastga_he.lca.legacy
            power_train_file_path: hybrid_propulsion.yml
            electric_mix: french

.. note::
    This override does not affect process we query directly from Ecoinvent, for instance the electricity used for the manufacturing of the battery, which we fetch directly as the :code:`'battery production, Li-ion, NMC111, rechargeable, prismatic'` process of EcoInvent, is not affected.

.. _normalization-options-lca:

Normalization
=============

As the normalization in a LCA analysis is an optional step, an option of the LCA has been added to enable it. This is done using the :code:`normalization` option in the configuration file. For this step, the normalization factor prescribed by the impact assessment method are used. For more information on those normalization factors, see the links to the method in the :ref:`impact-assessment-method-lca` subsection. Please note that if normalization factors are not available, as is the case for the Impact World+ method, this step is not carried out even if this option is set to :code:`True`.

.. code:: yaml

    model:
        lca:
            id: fastga_he.lca.legacy
            power_train_file_path: hybrid_propulsion.yml
            normalization: True

Weighting
=========

As for the :ref:`normalization <normalization-options-lca>` step, the weighting and aggregation step are optional. It can be enabled via the :code:`weighting` option. This step relies on the results from the normalization step, consequently in addition to the case where no weighting factors are available, this step won't be carried out if the normalization step hasn't been carried out.

.. code:: yaml

    model:
        lca:
            id: fastga_he.lca.legacy
            power_train_file_path: hybrid_propulsion.yml
            weighting: True

Use of operational mission
==========================

One of the key step in a LCA analysis is the choice of the functional unit. For aircraft, computation of the impacts per functional unit thus depends on the performances on a reference mission. By default, the LCA module in FAST-OAD-CS23-HE uses the sizing mission of the aircraft. An operational mission can alternatively be used by setting the :code:`use_operational_mission` option to :code:`True`.

.. code:: yaml

    model:
        lca:
            id: fastga_he.lca.legacy
            power_train_file_path: hybrid_propulsion.yml
            use_operational_mission: True

Definition of the aircraft lifespan and use
===========================================

As the LCA module computes the impact per functional unit (see the :ref:`models-lca` section for more information), the expected lifespan of the aircraft and its use are key information. By default, these data are inputted through the expected lifespan of the aircraft in years and its yearly number of flights. It can however be more convenient to input these data as an expected number of maximum airframe hours and the number of yearly hours flown. This can be enabled by setting the :code:`aircraft_lifespan_in_hours` option to :code:`True`.

.. code:: yaml

    model:
        lca:
            id: fastga_he.lca.legacy
            power_train_file_path: hybrid_propulsion.yml
            aircraft_lifespan_in_hours: True
