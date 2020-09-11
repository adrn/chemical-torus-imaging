# Third-party
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import gala.dynamics as gd

# This project
from thriftshop.potentials import galpy_potentials
from thriftshop.config import galcen_frame
from thriftshop.galpy_helpers import get_staeckel_aaf


def action_worker(task):
    (i1, i2), t, potential_name = task

    # Read APOGEE data
    c = coord.SkyCoord(ra=t['RA'] * u.deg,
                       dec=t['DEC'] * u.deg,
                       distance=1000 / t['GAIA_PARALLAX'] * u.pc,
                       pm_ra_cosdec=t['GAIA_PMRA'] * u.mas/u.yr,
                       pm_dec=t['GAIA_PMDEC'] * u.mas/u.yr,
                       radial_velocity=t['VHELIO_AVG'] * u.km/u.s)
    galcen = c.transform_to(galcen_frame)
    w0s = gd.PhaseSpacePosition(galcen.data)

    data = {'APOGEE_ID': [], 'actions': [], 'angles': [], 'freqs': []}
    aaf_units = {'actions': u.km/u.s*u.kpc, 'angles': u.rad, 'freqs': 1/u.Gyr}
    for n in range(w0s.shape[0]):
        apid = t['APOGEE_ID'][n]
        data['APOGEE_ID'].append(apid)

        # Use the Stäckel Fudge action solver:
        try:
            aaf = get_staeckel_aaf(w0s[n], galpy_potentials[potential_name])
        except Exception as e:
            print(f'failed to solve actions for {apid}:\n\t{str(e)}')
            for k in aaf_units.keys():
                data[k].append([np.nan] * 3 * aaf_units[k])
            continue

        for k in aaf_units.keys():
            data[k].append(aaf[k].to(aaf_units[k]))

    for k in aaf_units.keys():
        data[k] = u.Quantity(data[k])

    return data

