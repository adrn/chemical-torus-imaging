"""
Helper functions for computing actions using the axisymmetric Stäckel Fudge
method introduced in Binney 20XX, implemented in Galpy.
"""

# Standard library
import pickle

# Third-party
import astropy.coordinates as coord
import astropy.table as at
import astropy.units as u
import numpy as np
from scipy.interpolate import NearestNDInterpolator
from tqdm.auto import tqdm
from galpy.actionAngle import estimateDeltaStaeckel

# This project
from .config import rsun as ro, vcirc as vo, cache_path
from .potentials import galpy_potentials
from .galpy_helpers import gala_to_galpy_orbit


def get_staeckel_aaf(potential, w, delta=None):
    from galpy.actionAngle import estimateDeltaStaeckel, actionAngleStaeckel

    if delta is None:
        R = w.cylindrical.rho.to_value(ro)
        z = w.z.to_value(ro)
        delta = estimateDeltaStaeckel(potential, R, z)

    o = gala_to_galpy_orbit(w)
    aAS = actionAngleStaeckel(pot=potential, delta=delta)

    aaf = aAS.actionsFreqsAngles(o)
    aaf = {'actions': np.squeeze(aaf[:3]) * ro * vo,
           'freqs': np.squeeze(aaf[3:6]) * vo / ro,
           'angles': coord.Angle(np.squeeze(aaf[6:]) * u.rad)}

    return aaf


class StaeckelFudgeGrid:

    def __init__(self, Rz_grid_step=50*u.pc, Rz_grid_size=2.5*u.kpc):
        self._delta_interps = self._init_delta_interps(Rz_grid_step,
                                                       Rz_grid_size)

        self._delta_interps_keys = np.array(list(self._delta_interps.keys()))
        self._delta_interps_vals = np.array(
            [float(x) for x in self._delta_interps.keys()])

    def _get_delta_Rz_interp(self, galpy_potential,
                             grid_step=0.05*u.kpc, grid_size=2.5*u.kpc):

        grid_size = grid_size.to_value(u.kpc)
        grid_step = grid_step.to_value(u.kpc)

        # HACK: magic numbers
        Rz_grids = (np.arange(8-grid_size, 8+grid_size + 1e-5, grid_step),
                    np.arange(-grid_size, grid_size + 1e-5, grid_step))
        Rz_grid = np.stack(list(map(np.ravel, np.meshgrid(*Rz_grids)))).T

        delta_staeckels = []
        for i in range(Rz_grid.shape[0]):
            R = (Rz_grid[i, 0] * u.kpc).to_value(ro)
            z = (Rz_grid[i, 1] * u.kpc).to_value(ro)
            delta_staeckels.append(estimateDeltaStaeckel(
                galpy_potential, R, z))

        delta_staeckels = np.squeeze(delta_staeckels)

        delta_mask = np.isfinite(delta_staeckels)
        if delta_mask.sum() / len(delta_staeckels) < 0.9:
            print("Warning: More than 10% of the Staeckel ∆ values are bad.")

        delta_interp = NearestNDInterpolator(
            Rz_grid[delta_mask], delta_staeckels[delta_mask])

        return delta_interp

    def _init_delta_interps(self, Rz_grid_step, Rz_grid_size):
        path = cache_path / 'delta-interps.pkl'
        if path.exists():
            with open(path, 'rb') as f:
                delta_interps = pickle.load(f)
            return delta_interps

        print("Computing delta interpolators...")
        delta_interps = {}
        for k, pot in tqdm(galpy_potentials.items()):
            delta_interps[k] = self._get_delta_Rz_interp(pot,
                                                         grid_step=Rz_grid_step,
                                                         grid_size=Rz_grid_size)

        with open(path, 'wb') as f:
            pickle.dump(delta_interps, f)

        return delta_interps

    def get_aaf(self, w0, mdisk_val, galpy_potential):
        key = self._delta_interps_keys[
            np.abs(self._delta_interps_vals - mdisk_val).argmin()]
        delta_interp = self._delta_interps[key]
        deltas = delta_interp(w0.cylindrical.rho.to_value(u.kpc),
                              w0.z.to_value(u.kpc))

        aaf = get_staeckel_aaf(w0, galpy_potential, delta=deltas)
        return at.Table({k: v.T for k, v in aaf.items()})
