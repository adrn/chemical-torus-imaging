import astropy.units as u
import numpy as np
from scipy.optimize import minimize
import gala.potential as gp


potentials = dict()

# vcirc = 229 km/s @ solar circle (Eilers et al. 2019)
vcirc = 229.
rsun = 8.122
fidu_mdisk = 6.52551e10
potentials['fiducial'] = gp.MilkyWayPotential(disk=dict(m=fidu_mdisk))


def get_mw_potential(mdisk):
    """
    Retrieve a MW potential model with fixed vcirc=229
    """

    def objfunc(ln_mhalo):
        mhalo = np.exp(ln_mhalo)
        tmp_mw = gp.MilkyWayPotential(disk=dict(m=mdisk),
                                      halo=dict(m=mhalo))
        test_v = tmp_mw.circular_velocity([-rsun, 0, 0]).to_value(u.km/u.s)[0]
        return (vcirc - test_v) ** 2

    minit = potentials['fiducial']['halo'].parameters['m'].to_value(u.Msun)
    res = minimize(objfunc, x0=np.log(minit),
                   method='powell')

    if not res.success:
        return np.nan

    mhalo = np.exp(res.x)
    return gp.MilkyWayPotential(disk=dict(m=mdisk),
                                halo=dict(m=mhalo))


for fac in [0.4, 1.6]:
    potentials[f'{fac:.1f}'] = get_mw_potential(fac * fidu_mdisk)

for k, p in potentials.items():
    assert np.isclose(p.circular_velocity([-rsun, 0, 0]*u.kpc)[0],
                      vcirc * u.km/u.s,
                      rtol=1e-5)
