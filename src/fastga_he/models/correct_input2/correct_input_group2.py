import openmdao.api as om

import fastoad.api as oad
from .correct_name import HeNameFromRTAname2
from .compute_easy import EASY_compute2
from .set_value import SetValue2
from .correct_elevator import ComputeDeltaElevator2
from .correct_fus_cmalpha import ComputeCmAlphaFuselage2
from .aero_approximated import Aero_compute2


@oad.RegisterOpenMDAOSystem("fastga_he.correct_input2")
class SizingLoopMTOW(om.Group):
    """
    Gather all the discipline module/groups into the main problem
    """

    def setup(self):
        self.add_subsystem(name="correct_name", subsys=HeNameFromRTAname2(), promotes=["*"])
        self.add_subsystem(name="compute_easy", subsys=EASY_compute2(), promotes=["*"])
        self.add_subsystem(name="set_value", subsys=SetValue2(), promotes=["*"])
        self.add_subsystem(name="correct_elevator", subsys=ComputeDeltaElevator2(), promotes=["*"])
        self.add_subsystem(
            name="correct_fus_cmalpha", subsys=ComputeCmAlphaFuselage2(), promotes=["*"]
        )
        self.add_subsystem(name="aero_approximated", subsys=Aero_compute2(), promotes=["*"])
