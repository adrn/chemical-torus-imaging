# Standard library
import atexit
import os

# Third-party
from astropy.utils import iers
iers.conf.auto_download = False
import astropy.coordinates as coord
import astropy.table as at
import astropy.units as u
import numpy as np
import gala.dynamics as gd
import gala.integrate as gi

# This project
from thriftshop.potentials import potentials
from thriftshop.config import galcen_frame


def action_worker(task):
    (i1, i2), t, pot, potential_name, cache_path = task

    cache_filename = os.path.join(cache_path,
                                  f'tmp-{potential_name}-{i1}-{i2}.fits')

    print(f"Worker {i1} processing {len(t)} stars, saving to {cache_filename}")

    # Read APOGEE data
    coord.galactocentric_frame_defaults.set('v4.0')
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

        # First integrate a little bit of the orbit with Leapfrog to estimate
        # the orbital period
        test_orbit = pot.integrate_orbit(w0s[n], dt=2*u.Myr, t1=0, t2=1*u.Gyr)
        P_guess = test_orbit.estimate_period()

        if np.isnan(P_guess):
            print(f'failed to estimate period for {apid}')

            # Failed to estimate period - usually this means the data were very
            # noisy and the orbit was unbound or wacky
            for k in aaf_units.keys():
                data[k].append([np.nan] * 3 * aaf_units[k])
            continue

        # Integrate the orbit with a high-order integrator for many periods
        orbit = pot.integrate_orbit(w0s[n], dt=1*u.Myr, t1=0, t2=100 * P_guess,
                                    Integrator=gi.DOPRI853Integrator)

        # Use the Sanders & Binney action solver:
        try:
            aaf = gd.find_actions(orbit, N_max=8)  # N_max is a MAGIC NUMBER
        except Exception as e:
            print(f'failed to solve actions for {apid}:\n\t{str(e)}')
            for k in aaf_units.keys():
                data[k].append([np.nan] * 3 * aaf_units[k])
            continue

        for k in aaf_units.keys():
            data[k].append(aaf[k].to(aaf_units[k]))

    for k in aaf_units.keys():
        data[k] = u.Quantity(data[k])

    # Write results from this worker to tmp file
    tbl = at.Table(data)
    tbl.write(cache_filename)

    return cache_filename


def combine_output(all_filename, cache_path, potential_name):
    import glob

    cache_glob_pattr = os.path.join(cache_path,
                                    f'tmp-{potential_name}-*.fits')

    if os.path.exists(all_filename):
        prev_table = at.Table.read(all_filename)
    else:
        prev_table = None

    # combine the individual worker cache files
    all_tables = []
    remove_filenames = []
    for filename in glob.glob(cache_glob_pattr):
        all_tables.append(at.Table.read(filename))
        remove_filenames.append(filename)

    if all_tables:
        all_table = at.vstack(all_tables)
    else:
        return

    if prev_table:
        all_table = at.vstack((prev_table, all_table))

    all_table.write(all_filename, overwrite=True)

    for filename in remove_filenames:
        os.unlink(filename)


def main(pool, data_filename):
    from schwimmbad.utils import batch_tasks

    # Path to store generated datafiles
    cache_path = os.path.abspath(os.path.join(
        os.path.split(os.path.abspath(__file__))[0], '../cache'))
    os.makedirs(cache_path, exist_ok=True)

    # Register exit command for each potential
    all_filenames = {}
    for potential_name in potentials:
        all_filenames[potential_name] = os.path.join(
            cache_path, f'aaf-{potential_name}.fits')
        atexit.register(combine_output,
                        all_filenames[potential_name],
                        cache_path,
                        potential_name)

    # Load APOGEE data
    t = at.Table.read(data_filename)
    t = t[(t['GAIA_PARALLAX'] / t['GAIA_PARALLAX_ERROR']) > 5]

    # Execute worker on batches:
    for potential_name, potential in potentials.items():
        if os.path.exists(all_filenames[potential_name]):
            print(f"Actions exist for {potential_name} at "
                  f"{all_filenames[potential_name]}")
            continue

        tasks = batch_tasks(n_batches=max(1, pool.size - 1),
                            arr=t,
                            args=(potential, potential_name, cache_path))

        sub_filenames = []
        for sub_filename in pool.map(action_worker, tasks):
            sub_filenames.append(sub_filename)

        combine_output(all_filenames[potential_name],
                       cache_path,
                       potential_name)


if __name__ == '__main__':
    import sys
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser()

    parser.add_argument("--data", dest="data_filename", required=True,
                        type=str, help="the source data file")

    # vq_group = parser.add_mutually_exclusive_group()
    # vq_group.add_argument('-v', '--verbose', action='count', default=0,
    #                       dest='verbosity')
    # vq_group.add_argument('-q', '--quiet', action='count', default=0,
    #                       dest='quietness')

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--procs", dest="n_procs", default=1,
                       type=int, help="Number of processes.")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    parsed = parser.parse_args()

    # deal with multiproc:
    if parsed.mpi:
        from schwimmbad.mpi import MPIPool
        Pool = MPIPool
        kw = dict()
    elif parsed.n_procs > 1:
        from schwimmbad import MultiPool
        Pool = MultiPool
        kw = dict(processes=parsed.n_procs)
    else:
        from schwimmbad import SerialPool
        Pool = SerialPool
        kw = dict()
    Pool = Pool
    Pool_kwargs = kw

    with Pool(**Pool_kwargs) as pool:
        main(pool=pool, data_filename=parsed.data_filename)

    sys.exit(0)
