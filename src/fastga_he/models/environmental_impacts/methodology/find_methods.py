# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

try:
    import brightway2 as bw
    from dotenv import load_dotenv

    IMPORTS_LCA = True
except ImportError:
    IMPORTS_LCA = False

if __name__ == "__main__":
    if not IMPORTS_LCA:
        raise ImportError(
            "This feature is only usable with the lca optional dependency group.\n"
            "Install it with poetry install --extras lca"
        )

    bw.projects.set_current("MethodListing")
    load_dotenv()

    # if len(bw.databases) > 0:
    #     print("Initial setup already done, skipping")
    # else:
    #     # This is now the prefered method to init an Brightway2 with Ecoinvent
    #     # It is not more tied to a specific version of bw2io
    #     bw2io.import_ecoinvent_release(
    #         version="3.10",
    #         system_model="cutoff",
    #         username=os.environ["ECOINVENT_LOGIN"],  # Read for .env file
    #         password=os.environ["ECOINVENT_PASSWORD"],  # Read from .env file
    #         use_mp=True)

    method_list = list(bw.methods)

    if not bw.methods:
        print("Databases were not loaded, please uncomment the block above")

    recipe = "ReCiPe 2016 v1.03, midpoint (H)"
    environmental_footprint = "EF v3.1"
    impact_world = "IMPACT World+ v2.0.1, footprint version"

    recipe_methods = []
    ef_method = []
    iw_method = []

    for method in method_list:
        if recipe in method[0]:
            recipe_methods.append(method)
        elif environmental_footprint in method[0]:
            ef_method.append(method)
        elif impact_world in method[0]:
            iw_method.append(method)
        else:
            test = 1

    print(recipe_methods, end="\n")
    print(ef_method, end="\n")
    print(iw_method, end="\n")

    recipe_methods_formatted = [
        (
            "ReCiPe 2016 v1.03, midpoint (H) no LT",
            "acidification: terrestrial no LT",
            "terrestrial acidification potential (TAP) no LT",
        ),
        (
            "ReCiPe 2016 v1.03, midpoint (H) no LT",
            "climate change no LT",
            "global warming potential (GWP1000) no LT",
        ),
        (
            "ReCiPe 2016 v1.03, midpoint (H) no LT",
            "ecotoxicity: freshwater no LT",
            "freshwater ecotoxicity potential (FETP) no LT",
        ),
        (
            "ReCiPe 2016 v1.03, midpoint (H) no LT",
            "ecotoxicity: marine no LT",
            "marine ecotoxicity potential (METP) no LT",
        ),
        (
            "ReCiPe 2016 v1.03, midpoint (H) no LT",
            "ecotoxicity: terrestrial no LT",
            "terrestrial ecotoxicity potential (TETP) no LT",
        ),
        (
            "ReCiPe 2016 v1.03, midpoint (H) no LT",
            "energy resources: non-renewable, fossil no LT",
            "fossil fuel potential (FFP) no LT",
        ),
        (
            "ReCiPe 2016 v1.03, midpoint (H) no LT",
            "eutrophication: freshwater no LT",
            "freshwater eutrophication potential (FEP) no LT",
        ),
        (
            "ReCiPe 2016 v1.03, midpoint (H) no LT",
            "eutrophication: marine no LT",
            "marine eutrophication potential (MEP) no LT",
        ),
        (
            "ReCiPe 2016 v1.03, midpoint (H) no LT",
            "human toxicity: carcinogenic no LT",
            "human toxicity potential (HTPc) no LT",
        ),
        (
            "ReCiPe 2016 v1.03, midpoint (H) no LT",
            "human toxicity: non-carcinogenic no LT",
            "human toxicity potential (HTPnc) no LT",
        ),
        (
            "ReCiPe 2016 v1.03, midpoint (H) no LT",
            "ionising radiation no LT",
            "ionising radiation potential (IRP) no LT",
        ),
        (
            "ReCiPe 2016 v1.03, midpoint (H) no LT",
            "land use no LT",
            "agricultural land occupation (LOP) no LT",
        ),
        (
            "ReCiPe 2016 v1.03, midpoint (H) no LT",
            "material resources: metals/minerals no LT",
            "surplus ore potential (SOP) no LT",
        ),
        (
            "ReCiPe 2016 v1.03, midpoint (H) no LT",
            "ozone depletion no LT",
            "ozone depletion potential (ODPinfinite) no LT",
        ),
        (
            "ReCiPe 2016 v1.03, midpoint (H) no LT",
            "particulate matter formation no LT",
            "particulate matter formation potential (PMFP) no LT",
        ),
        (
            "ReCiPe 2016 v1.03, midpoint (H) no LT",
            "photochemical oxidant formation: human health no LT",
            "photochemical oxidant formation potential: humans (HOFP) no LT",
        ),
        (
            "ReCiPe 2016 v1.03, midpoint (H) no LT",
            "photochemical oxidant formation: terrestrial ecosystems no LT",
            "photochemical oxidant formation potential: ecosystems (EOFP) no LT",
        ),
        (
            "ReCiPe 2016 v1.03, midpoint (H) no LT",
            "water use no LT",
            "water consumption potential (WCP) no LT",
        ),
        (
            "ReCiPe 2016 v1.03, midpoint (H)",
            "acidification: terrestrial",
            "terrestrial acidification potential (TAP)",
        ),
        ("ReCiPe 2016 v1.03, midpoint (H)", "climate change", "global warming potential (GWP1000)"),
        (
            "ReCiPe 2016 v1.03, midpoint (H)",
            "ecotoxicity: freshwater",
            "freshwater ecotoxicity potential (FETP)",
        ),
        (
            "ReCiPe 2016 v1.03, midpoint (H)",
            "ecotoxicity: marine",
            "marine ecotoxicity potential (METP)",
        ),
        (
            "ReCiPe 2016 v1.03, midpoint (H)",
            "ecotoxicity: terrestrial",
            "terrestrial ecotoxicity potential (TETP)",
        ),
        (
            "ReCiPe 2016 v1.03, midpoint (H)",
            "energy resources: non-renewable, fossil",
            "fossil fuel potential (FFP)",
        ),
        (
            "ReCiPe 2016 v1.03, midpoint (H)",
            "eutrophication: freshwater",
            "freshwater eutrophication potential (FEP)",
        ),
        (
            "ReCiPe 2016 v1.03, midpoint (H)",
            "eutrophication: marine",
            "marine eutrophication potential (MEP)",
        ),
        (
            "ReCiPe 2016 v1.03, midpoint (H)",
            "human toxicity: carcinogenic",
            "human toxicity potential (HTPc)",
        ),
        (
            "ReCiPe 2016 v1.03, midpoint (H)",
            "human toxicity: non-carcinogenic",
            "human toxicity potential (HTPnc)",
        ),
        (
            "ReCiPe 2016 v1.03, midpoint (H)",
            "ionising radiation",
            "ionising radiation potential (IRP)",
        ),
        ("ReCiPe 2016 v1.03, midpoint (H)", "land use", "agricultural land occupation (LOP)"),
        (
            "ReCiPe 2016 v1.03, midpoint (H)",
            "material resources: metals/minerals",
            "surplus ore potential (SOP)",
        ),
        (
            "ReCiPe 2016 v1.03, midpoint (H)",
            "ozone depletion",
            "ozone depletion potential (ODPinfinite)",
        ),
        (
            "ReCiPe 2016 v1.03, midpoint (H)",
            "particulate matter formation",
            "particulate matter formation potential (PMFP)",
        ),
        (
            "ReCiPe 2016 v1.03, midpoint (H)",
            "photochemical oxidant formation: human health",
            "photochemical oxidant formation potential: humans (HOFP)",
        ),
        (
            "ReCiPe 2016 v1.03, midpoint (H)",
            "photochemical oxidant formation: terrestrial ecosystems",
            "photochemical oxidant formation potential: ecosystems (EOFP)",
        ),
        ("ReCiPe 2016 v1.03, midpoint (H)", "water use", "water consumption potential (WCP)"),
        (
            "ReCiPe 2016 v1.03, midpoint (H) no LT",
            "climate change no LT",
            "global warming potential (GWP100) no LT",
        ),
        ("ReCiPe 2016 v1.03, midpoint (H)", "climate change", "global warming potential (GWP100)"),
    ]

    ef_no_lt_method_formatted = [
        ("EF v3.1 no LT", "acidification no LT", "accumulated exceedance (AE) no LT"),
        ("EF v3.1 no LT", "climate change no LT", "global warming potential (GWP100) no LT"),
        (
            "EF v3.1 no LT",
            "climate change: biogenic no LT",
            "global warming potential (GWP100) no LT",
        ),
        (
            "EF v3.1 no LT",
            "climate change: fossil no LT",
            "global warming potential (GWP100) no LT",
        ),
        (
            "EF v3.1 no LT",
            "climate change: land use and land use change no LT",
            "global warming potential (GWP100) no LT",
        ),
        (
            "EF v3.1 no LT",
            "ecotoxicity: freshwater no LT",
            "comparative toxic unit for ecosystems (CTUe) no LT",
        ),
        (
            "EF v3.1 no LT",
            "ecotoxicity: freshwater, inorganics no LT",
            "comparative toxic unit for ecosystems (CTUe) no LT",
        ),
        (
            "EF v3.1 no LT",
            "ecotoxicity: freshwater, organics no LT",
            "comparative toxic unit for ecosystems (CTUe) no LT",
        ),
        (
            "EF v3.1 no LT",
            "energy resources: non-renewable no LT",
            "abiotic depletion potential (ADP): fossil fuels no LT",
        ),
        (
            "EF v3.1 no LT",
            "eutrophication: freshwater no LT",
            "fraction of nutrients reaching freshwater end compartment (P) no LT",
        ),
        (
            "EF v3.1 no LT",
            "eutrophication: marine no LT",
            "fraction of nutrients reaching marine end compartment (N) no LT",
        ),
        (
            "EF v3.1 no LT",
            "eutrophication: terrestrial no LT",
            "accumulated exceedance (AE) no LT",
        ),
        (
            "EF v3.1 no LT",
            "human toxicity: carcinogenic no LT",
            "comparative toxic unit for human (CTUh) no LT",
        ),
        (
            "EF v3.1 no LT",
            "human toxicity: carcinogenic, inorganics no LT",
            "comparative toxic unit for human (CTUh) no LT",
        ),
        (
            "EF v3.1 no LT",
            "human toxicity: carcinogenic, organics no LT",
            "comparative toxic unit for human (CTUh) no LT",
        ),
        (
            "EF v3.1 no LT",
            "human toxicity: non-carcinogenic no LT",
            "comparative toxic unit for human (CTUh) no LT",
        ),
        (
            "EF v3.1 no LT",
            "human toxicity: non-carcinogenic, inorganics no LT",
            "comparative toxic unit for human (CTUh) no LT",
        ),
        (
            "EF v3.1 no LT",
            "human toxicity: non-carcinogenic, organics no LT",
            "comparative toxic unit for human (CTUh) no LT",
        ),
        (
            "EF v3.1 no LT",
            "ionising radiation: human health no LT",
            "human exposure efficiency relative to u235 no LT",
        ),
        ("EF v3.1 no LT", "land use no LT", "soil quality index no LT"),
        (
            "EF v3.1 no LT",
            "material resources: metals/minerals no LT",
            "abiotic depletion potential (ADP): elements (ultimate reserves) no LT",
        ),
        ("EF v3.1 no LT", "ozone depletion no LT", "ozone depletion potential (ODP) no LT"),
        ("EF v3.1 no LT", "particulate matter formation no LT", "impact on human health no LT"),
        (
            "EF v3.1 no LT",
            "photochemical oxidant formation: human health no LT",
            "tropospheric ozone concentration increase no LT",
        ),
        (
            "EF v3.1 no LT",
            "water use no LT",
            "user deprivation potential (deprivation-weighted water consumption) no LT",
        ),
    ]

    ef_methods_formatted = [
        ("EF v3.1", "acidification", "accumulated exceedance (AE)"),
        ("EF v3.1", "climate change", "global warming potential (GWP100)"),
        ("EF v3.1", "climate change: biogenic", "global warming potential (GWP100)"),
        ("EF v3.1", "climate change: fossil", "global warming potential (GWP100)"),
        (
            "EF v3.1",
            "climate change: land use and land use change",
            "global warming potential (GWP100)",
        ),
        ("EF v3.1", "ecotoxicity: freshwater", "comparative toxic unit for ecosystems (CTUe)"),
        (
            "EF v3.1",
            "ecotoxicity: freshwater, inorganics",
            "comparative toxic unit for ecosystems (CTUe)",
        ),
        (
            "EF v3.1",
            "ecotoxicity: freshwater, organics",
            "comparative toxic unit for ecosystems (CTUe)",
        ),
        (
            "EF v3.1",
            "energy resources: non-renewable",
            "abiotic depletion potential (ADP): fossil fuels",
        ),
        (
            "EF v3.1",
            "eutrophication: freshwater",
            "fraction of nutrients reaching freshwater end compartment (P)",
        ),
        (
            "EF v3.1",
            "eutrophication: marine",
            "fraction of nutrients reaching marine end compartment (N)",
        ),
        ("EF v3.1", "eutrophication: terrestrial", "accumulated exceedance (AE)"),
        ("EF v3.1", "human toxicity: carcinogenic", "comparative toxic unit for human (CTUh)"),
        (
            "EF v3.1",
            "human toxicity: carcinogenic, inorganics",
            "comparative toxic unit for human (CTUh)",
        ),
        (
            "EF v3.1",
            "human toxicity: carcinogenic, organics",
            "comparative toxic unit for human (CTUh)",
        ),
        (
            "EF v3.1",
            "human toxicity: non-carcinogenic",
            "comparative toxic unit for human (CTUh)",
        ),
        (
            "EF v3.1",
            "human toxicity: non-carcinogenic, inorganics",
            "comparative toxic unit for human (CTUh)",
        ),
        (
            "EF v3.1",
            "human toxicity: non-carcinogenic, organics",
            "comparative toxic unit for human (CTUh)",
        ),
        (
            "EF v3.1",
            "ionising radiation: human health",
            "human exposure efficiency relative to u235",
        ),
        ("EF v3.1", "land use", "soil quality index"),
        (
            "EF v3.1",
            "material resources: metals/minerals",
            "abiotic depletion potential (ADP): elements (ultimate reserves)",
        ),
        ("EF v3.1", "ozone depletion", "ozone depletion potential (ODP)"),
        ("EF v3.1", "particulate matter formation", "impact on human health"),
        (
            "EF v3.1",
            "photochemical oxidant formation: human health",
            "tropospheric ozone concentration increase",
        ),
        (
            "EF v3.1",
            "water use",
            "user deprivation potential (deprivation-weighted water consumption)",
        ),
    ]
    iw_method_formatted = [
        ("IMPACT World+ v2.0.1, footprint version", "climate change", "carbon footprint"),
        (
            "IMPACT World+ v2.0.1, footprint version",
            "ecosystem quality",
            "remaining ecosystem quality damage",
        ),
        (
            "IMPACT World+ v2.0.1, footprint version",
            "energy resources: non-renewable",
            "fossil and nuclear energy use",
        ),
        (
            "IMPACT World+ v2.0.1, footprint version",
            "human health",
            "remaining human health damage",
        ),
        ("IMPACT World+ v2.0.1, footprint version", "water use", "water scarcity footprint"),
    ]
