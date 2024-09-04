import numpy as np
import openmdao.api as om


class ComputeGeometry(om.Group):
    def setup(self):
        self.add_subsystem(name="compute_wing_area", subsys=ComputeWingArea(), promotes=["*"])


class ComputeWingArea(om.ExplicitComponent):
    """
    Computes the wing area based on the provided MTOW and wing loading
    """

    def setup(self):
        # Defining the input(s)

        self.add_input(name="mtow", units="kg", val=np.nan)
        self.add_input(name="wing_loading", units="kg/m**2", val=np.nan)

        # Defining the output(s)

        self.add_output(name="wing_area", units="m**2")
        self.add_output(name="dummy_vector", units="m**2", shape=3)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Assigning the input to local variable for clarity
        mtow = inputs["mtow"]

        wing_loading = inputs["wing_loading"]

        # Computation of the wing area
        wing_area = mtow / wing_loading

        outputs["wing_area"] = wing_area
        outputs["dummy_vector"] = wing_area * np.array([1.0, 2.0, 3.0])
