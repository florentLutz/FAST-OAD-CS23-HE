.. _assumptions-lca:

=======================================
Life Cycle Assessment model assumptions
=======================================

Scope of the LCA analysis
=========================

The LCA covers the production of the components, the assembly of the airframe, the testing of the aircraft before acceptance, its distribution to the end user and the use phase. The use phase includes the generation of electricity, the production of fuel and its combustion during flight. Ground infrastructure (e.g. airfields) and end-of-life treatment of the aircraft and its components are excluded from the study, although future work could investigate the potential impact of these omissions.

A nomenclature similar to that used in early drafts of the aircraft PEFCR has been adopted in the data structures and post-processing graphs. The following categories and their sub-processes are listed here:

* Production: production of airframe components, airframe assembly and powertrain components
* Manufacturing: acceptance tests.
* Distribution: distribution according to one of two selected modes, see the section on the :ref:`options-lca`.
* Operation: combustion and production of fuels and generation of electricity.

Because of their relative simplicity and small contribution to the final impact, manufacturing and distribution have been assumed to be proportionate to the operation phase. Therefore, all their sub-processes have been aggregated, unlike the other phases for which a breakdown of contribution is available.

Assumptions regarding assembly of components
============================================

While the impacts of most components is computed using proxies from the EcoInvent database, for the others, inventories were reconstructed from literature. Data for those reconstructed inventories should include the production of materials and other inputs required for the assembly of the components. For the former, data from :cite:`thonemann:2024` are used, and while data from the latter is also available it seems like the values given might already include the production of the materials. Consequently, and to avoid counting effect twice, for components with no proxies in the EcoInvent database, assembly will be discarded and only impacts from the production of materials will be considered.

Other assumptions
=================

This is a Work-In-Progress, future version of the documentation will include a description of:

* The assumptions made on the materials of the landing gear
* The assumptions made on the electric mix
* The proxies used