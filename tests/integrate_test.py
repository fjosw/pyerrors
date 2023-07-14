import numpy as np
import autograd.numpy as anp
import pyerrors as pe


def test_integration():
    def f(p, x):
        return p[0] * x + p[1] * x**2 - p[2] / x

    def F(p, x):
        return p[0] * x**2 / 2. + p[1] * x**3 / 3. - anp.log(x) * p[2]

    def check_ana_vs_int(p, l, u, **kwargs):
        numint_full = pe.integrate.quad(f, p, l, u, **kwargs)
        numint = numint_full[0]
        if isinstance(numint, pe.Obs):
            numint.gm()

        anaint = F(p, u) - F(p, l)
        anaint.gm()

        diff = (numint - anaint)
        assert(diff.is_zero())

    pobs = np.array([pe.cov_Obs(1., .1**2, '0'), pe.cov_Obs(2., .2**2, '1'), pe.cov_Obs(2.2, .17**2, '2')])
    lobs = pe.cov_Obs(.123, .012**2, 'l')
    uobs = pe.cov_Obs(1., .05**2, 'u')

    check_ana_vs_int(pobs, lobs, uobs)
    check_ana_vs_int(pobs, lobs.value, uobs)
    check_ana_vs_int(pobs, lobs, uobs.value)
    check_ana_vs_int(pobs, lobs.value, uobs.value)
    for i in range(len(pobs)):
        p = [pi for pi in pobs]
        p[i] = pobs[i].value
        check_ana_vs_int(p, lobs, uobs)

    check_ana_vs_int([pi.value for pi in pobs], lobs, uobs)

    check_ana_vs_int(pobs, lobs, uobs, epsabs=1.e-9, epsrel=1.236e-10, limit=100)
    assert(len(pe.integrate.quad(f, pobs, lobs, uobs, full_output=True)) > 2)

    r1, _ = pe.integrate.quad(F, pobs, 1, 0.1)
    r2, _ = pe.integrate.quad(F, pobs, 0.1, 1)
    assert r1 == -r2
    iamzero, _ = pe.integrate.quad(F, pobs, 1, 1)
    assert iamzero == 0
