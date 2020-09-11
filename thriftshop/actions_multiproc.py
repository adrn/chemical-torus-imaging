# Third-party
import astropy.coordinates as coord
import astropy.table as at
import astropy.units as u
import numpy as np
import gala.dynamics as gd

# This project
from thriftshop.potentials import galpy_potentials
from thriftshop.config import galcen_frame
from thriftshop.galpy_helpers import get_staeckel_aaf


def action_worker(task):
    (i1, i2), w0s, apogee_ids, potential_name = task

    data = {'APOGEE_ID': [], 'actions': [], 'angles': [], 'freqs': []}
    aaf_units = {'actions': u.km/u.s*u.kpc, 'angles': u.rad, 'freqs': 1/u.Gyr}
    for n in range(w0s.shape[0]):
        data['APOGEE_ID'].append(apogee_ids[n])

        # Use the Stäckel Fudge action solver:
        try:
            aaf = get_staeckel_aaf(w0s[n], galpy_potentials[potential_name])
        except Exception as e:
            print(f'failed to solve actions for {apogee_ids[n]}:\n\t{str(e)}')
            for k in aaf_units.keys():
                data[k].append([np.nan] * 3 * aaf_units[k])
            continue

        for k in aaf_units.keys():
            data[k].append(aaf[k].to(aaf_units[k]))

    for k in aaf_units.keys():
        data[k] = u.Quantity(data[k])

    return at.QTable(data)


def compute_actions_multiproc(t, potential_name, pool):
    from schwimmbad.utils import batch_tasks

    c = coord.SkyCoord(ra=t['RA'] * u.deg,
                       dec=t['DEC'] * u.deg,
                       distance=1000 / t['GAIA_PARALLAX'] * u.pc,
                       pm_ra_cosdec=t['GAIA_PMRA'] * u.mas/u.yr,
                       pm_dec=t['GAIA_PMDEC'] * u.mas/u.yr,
                       radial_velocity=t['VHELIO_AVG'] * u.km/u.s)
    galcen = c.transform_to(galcen_frame)
    w0s = gd.PhaseSpacePosition(galcen.data)
    apogee_ids = t['APOGEE_ID']

    tasks_idx = batch_tasks(n_batches=max(1, pool.size),
                            n_tasks=len(t))
    tasks = [((i1, i2), w0s[i1:i2], apogee_ids[i1:i2], potential_name)
             for i1, i2 in tasks_idx]

    results = []
    for result in pool.map(action_worker, tasks):
        results.append(result)

    return at.vstack(results)
