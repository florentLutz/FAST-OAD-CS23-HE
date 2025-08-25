import openmdao.api as om

import fastoad.api as oad
from .correct_name import HeNameFromRTAname
from .compute_easy import EASY_compute
from .set_value import SetValue
from .correct_elevator import ComputeDeltaElevator
from .correct_fus_cmalpha import ComputeCmAlphaFuselage
from .aero_approximated import Aero_compute


@oad.RegisterOpenMDAOSystem("fastga_he.correct_input")
class SizingLoopMTOW(om.Group):
    """
    Gather all the discipline module into the main problem
    """

    def setup(self):
        self.add_subsystem(name="correct_name", subsys=HeNameFromRTAname(), promotes=["*"])
        self.add_subsystem(name="compute_easy", subsys=EASY_compute(), promotes=["*"])
        self.add_subsystem(name="set_value", subsys=SetValue(), promotes=["*"])
        self.add_subsystem(name="correct_elevator", subsys=ComputeDeltaElevator(), promotes=["*"])
        self.add_subsystem(
            name="correct_fus_cmalpha", subsys=ComputeCmAlphaFuselage(), promotes=["*"]
        )
        self.add_subsystem(name="aero_approximated", subsys=Aero_compute(), promotes=["*"])
