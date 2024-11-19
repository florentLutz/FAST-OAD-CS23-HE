# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO


import openmdao.api as om


class EmissionsRenamer(om.ExplicitComponent):
    """
    Simple utility class to rename emissions from sizing to operational
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.renamer_dict = None

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        for inputs_to_rename in self.renamer_dict.keys():
            outputs[self.renamer_dict[inputs_to_rename]] = inputs[inputs_to_rename]
