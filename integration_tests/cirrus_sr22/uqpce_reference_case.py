import numpy as np
from uqpce.pce.pce import PCE

samp_count = 50
aleat_cnt = 500000
epist_cnt = 1

def test_uqpce():
    pce = PCE(
        order=2, verbose=True, outputs=True, plot=False, aleat_samp_size=aleat_cnt,
        epist_samp_size=epist_cnt
    )

    # Add two normal variables
    pce.add_variable(distribution='normal', mean=1, stdev=3, name='uncerta')
    pce.add_variable(distribution='normal', mean=1, stdev=7, name='uncertb')

    # Generate samples that correspond to the input variables
    Xt = pce.sample(count=samp_count)

    # Generate responses from equation; the user's analytical tool will replace this
    eq = 'x0**2 + x0*x1 - x1'
    yt = pce.generate_responses(Xt, equation=eq)

    print(Xt)
    print(yt)

    pce.fit(np.array(Xt), np.array(yt)) # Fit the PCE model
    pce.check_variables(np.array(Xt)) # Check if the samples correspond to the distributions
    pce.sobols() # Calculate the Sobol indices
    cil, cih = pce.confidence_interval() # Calculate the confidence interval
    pce.write_outputs()