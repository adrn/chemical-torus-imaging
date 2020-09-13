# Standard library
import sys

# Third-party
from astropy.utils import iers
iers.conf.auto_download = False
import astropy.table as at
import numpy as np

# This project
from thriftshop.config import cache_path, elem_names
from thriftshop.data import load_apogee_sample
from thriftshop.objective import TorusImagingObjective


def worker(task):
    i, obj, x0 = task
    try:
        res = obj.minimize(x0=x0)
    except Exception as e:
        print(f"{i} failed: {str(e)}")
        return None

    if not res.success:
        return np.nan * res.x

    else:
        return res.x


def main(pool, overwrite=False):
    tree_K = 20  # MAGIC NUMBER: set heuristically in Objective-function.ipynb
    bootstrap_K = 128  # MAGIC NUMBER

    t, c = load_apogee_sample('../data/apogee-parent-sample.fits')

    # TODO: make seed configurable?
    rnd = np.random.default_rng(seed=42)

    # TODO: loop over all elements
    # for elem_name in elem_names:
    for elem_name in ['MG_FE']:
        print(f"Running element: {elem_name}")

        this_cache_filename = cache_path / f'optimize-results-{elem_name}.csv'
        if this_cache_filename.exists() and not overwrite:
            print(f"Cache file exists for {elem_name}: {this_cache_filename}")
            continue

        print("Optimizing with full sample to initialize bootstraps...")
        obj = TorusImagingObjective(t, c, elem_name, tree_K=tree_K)
        full_sample_res = obj.minimize()
        if not full_sample_res.success:
            print(f"FAILED TO CONVERGE: optimize for full sample failed for "
                  f"{elem_name}")
            continue

        print(f"Finished optimizing full sample: {full_sample_res.x}")

        tasks = []
        for k in range(bootstrap_K):
            idx = rnd.choice(len(t), len(t), replace=True)
            obj = TorusImagingObjective(t[idx], c[idx], elem_name,
                                        tree_K=tree_K)
            tasks.append((k, obj, full_sample_res.x))

        print("Done setting up bootstrap samples - running pool.map() ...")

        results = []
        for res in pool.map(worker, tasks):
            results.append(res)
        results = np.array([x for x in results if x is not None])
        results = at.Table({
            'mdisk_f': results[:, 0],
            'zsun': results[:, 1],
            'vzsun': results[:, 2]
        })
        results.write(this_cache_filename, overwrite=True)

    sys.exit(0)


if __name__ == '__main__':
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser()

    parser.add_argument("-o", "--overwrite", dest="overwrite",
                        action="store_true")

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
        main(pool=pool, overwrite=parsed.overwrite)

    sys.exit(0)
