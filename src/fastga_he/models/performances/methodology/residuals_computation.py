# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth

import numpy as np
import openmdao.api as om

from .modules.mtow_loop import SizingLoopMTOW

RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")

if __name__ == "__main__":
    input_ivc = om.IndepVarComp()

    input_ivc.add_output(name="wing_loading", val=115.0, units="kg/m**2")
    input_ivc.add_output(name="aspect_ratio", val=10.0)

    input_ivc.add_output(name="cruise_altitude", val=2500.0, units="m")
    input_ivc.add_output(name="cruise_speed", val=80.0, units="m/s")
    input_ivc.add_output(name="mission_range", val=750, units="NM")
    input_ivc.add_output(name="payload", val=320, units="kg")

    input_ivc.add_output(name="tsfc", val=7.3e-6, units="kg/N/s")

    prob = om.Problem(reports=False)
    prob.model.add_subsystem(name="inputs_definition", subsys=input_ivc, promotes=["*"])
    prob.model.add_subsystem(name="mtow_solving", subsys=SizingLoopMTOW(), promotes=["*"])

    prob.model.nonlinear_solver = om.NonlinearBlockGS()
    prob.model.linear_solver = om.LinearBlockGS()

    recorder = om.SqliteRecorder(pth.join(RESULTS_FOLDER_PATH, "cases.sql"))
    prob.model.nonlinear_solver.add_recorder(recorder)
    prob.model.nonlinear_solver.recording_options["record_solver_residuals"] = True
    prob.model.nonlinear_solver.recording_options["record_outputs"] = True

    prob.model.nonlinear_solver.options["maxiter"] = 50
    prob.model.nonlinear_solver.options["iprint"] = 2
    prob.model.nonlinear_solver.options["atol"] = 1e-4

    prob.setup()

    prob.set_val(name="mtow", val=500.0, units="kg")

    prob.run_model()

    print(
        "The solution MTOW for the problem with IVC is :",
        float(np.round(prob.get_val("mtow", units="kg"), 1)),
        "kg",
    )

    recorder_data_file_path = pth.join(RESULTS_FOLDER_PATH, "cases.sql")

    cr = om.CaseReader(recorder_data_file_path)

    cases = cr.get_cases("root.nonlinear_solver")

    for case in cases:
        output_to_print = []
        residuals_to_print = []

        for residual in case.residuals:
            if any(case.residuals[residual]) != 0:
                value = case.residuals[residual]

                if len(value) == 1:
                    residuals_to_print.append(float(value))
                    output_to_print.append(float(case.outputs[residual]))

                else:
                    for val, out in zip(value, case.outputs[residual]):
                        residuals_to_print.append(float(val))
                        output_to_print.append(float(out))

        print(residuals_to_print)
        print(output_to_print)
