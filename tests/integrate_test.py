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

        anaint = F(p, u) - F(p, l)
        diff = (numint - anaint)

        if isinstance(numint, pe.Obs):
            numint.gm()
            anaint.gm()

            assert(diff.is_zero())
        else:
            assert(np.isclose(0, diff))

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
    check_ana_vs_int([pi.value for pi in pobs], lobs.value, uobs.value)

    check_ana_vs_int(pobs, lobs, uobs, epsabs=1.e-9, epsrel=1.236e-10, limit=100)
    assert(len(pe.integrate.quad(f, pobs, lobs, uobs, full_output=True)) > 2)

    r1, _ = pe.integrate.quad(F, pobs, 1, 0.1)
    r2, _ = pe.integrate.quad(F, pobs, 0.1, 1)
    assert r1 == -r2
    iamzero, _ = pe.integrate.quad(F, pobs, 1, 1)
    assert iamzero == 0


def test_integrate_per_parameter_derivatives():
    # \int_0^1 p0*x + p1*x^2 dx = p0/2 + p1/3
    # If the lambda closure in integrate.quad failed to bind `i` per-iteration,
    # all per-parameter derivatives would collapse to a single value.
    def f(p, x):
        return p[0] * x + p[1] * x ** 2

    p = [pe.cov_Obs(1.0, 0.1 ** 2, "p0"), pe.cov_Obs(2.0, 0.2 ** 2, "p1")]
    res, _ = pe.integrate.quad(f, p, 0.0, 1.0)
    res.gm()

    ana_val = p[0].value / 2 + p[1].value / 3
    assert np.isclose(res.value, ana_val)

    grad0 = res.covobs["p0"].grad.item()
    grad1 = res.covobs["p1"].grad.item()
    assert np.isclose(grad0, 0.5)
    assert np.isclose(grad1, 1.0 / 3.0)
    assert not np.isclose(grad0, grad1)
