import astropy.units as u
import numpy as np
from .config import rsun, vcirc

ro = rsun
vo = vcirc


def gala_to_galpy_orbit(w):
    from galpy.orbit import Orbit

    # PhaseSpacePosition or Orbit:
    cyl = w.cylindrical

    R = cyl.rho.to_value(ro).T
    phi = cyl.phi.to_value(u.rad).T
    z = cyl.z.to_value(ro).T

    vR = cyl.v_rho.to_value(vo).T
    vT = (cyl.rho * cyl.pm_phi).to_value(vo, u.dimensionless_angles()).T
    vz = cyl.v_z.to_value(vo).T

    o = Orbit(np.array([R, vR, vT, z, vz, phi]).T, ro=ro, vo=vo)

    if hasattr(w, 't'):
        o.t = w.t.to_value(u.Myr)

    return o


def get_staeckel_actions(w, potential):
    from galpy.actionAngle import estimateDeltaStaeckel, actionAngleStaeckel

    R = w.cylindrical.rho.to_value(ro)
    z = w.z.to_value(ro)
    delta = estimateDeltaStaeckel(potential, R, z)

    o = gala_to_galpy_orbit(w)
    aAS = actionAngleStaeckel(pot=potential, delta=delta)
    return np.array(aAS(o)).squeeze() * ro * vo
