# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

METHODS_TO_FILE = {
    "EF v3.1 no LT": "ef_no_lt_methods.yml",
    "EF v3.1": "ef_methods.yml",
    "IMPACT World+ v2.0.1": "impact_world_methods.yml",
    "ReCiPe 2016 v1.03": "recipe_methods.yml",
}
METHODS_TO_NORMALIZATION = {
    "EF v3.1 no LT": True,
    "EF v3.1": True,
    "IMPACT World+ v2.0.1": True,
    "ReCiPe 2016 v1.03": True,
}
NORMALIZATION_FACTOR = {
    "EF v3.1 no LT":
        {
            "acidification_terrestrial": 5.56e01,
            "climate_change": 7.55e03,
            "ecotoxicity_freshwater": 5.67e04,
            "ecotoxicity_marine": 1,  # ???
            "ecotoxicity_terrestrial": 1,  # ???
            "energy_resources_non-renewablefossil": 6.50e04,
            "eutrophication_freshwater": 1.61e00,
            "eutrophication_marine": 1.95e01,
            "human_toxicity_carcinogenic": 1.73e-05,
            "human_toxicity_non-carcinogenic": 1.29e-04,
            "ionising_radiation": 4.22e03,
            "land_use": 8.19e05,
            "material_resources_metals_minerals": 6.36e-02,
            "ozone_depletion": 5.23e-02,
            "particulate_matter_formation": 5.95e-04,
            "photochemical_oxidant_formation_human_health": 1,  # ???
            "photochemical_oxidant_formation_terrestrial_ecosystems": 1,  # ???
            "total_ecosystem_quality": 1,  # ???
            "total_human_health": 1,  # ???
            "total_natural_resources": 1,  # ???
            "water_use": 1.15e04,
        }
}
