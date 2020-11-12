# Standard library
import atexit
import os
import sys

# Third-party
from astropy.utils import iers
iers.conf.auto_download = False
import astropy.table as at
import numpy as np

# This project
from totoro.config import cache_path
from totoro.data import datasets, elem_names
from totoro.objective import TorusImagingObjective


def worker(task):
    i, obj, x0, tmp_filename = task

    res = None
    try:
        res = obj.minimize(x0=x0, method="nelder-mead",
                           options=dict(maxiter=250))
        print(f"{i} finished optimizing: {res}")
    except Exception as e:
        print(f"{i} failed: {str(e)}")

    if res is None or not res.success:
        xopt = np.nan * np.array(x0)
    else:
        xopt = res.x

    xopt = {
        'zsun': [xopt[0]],
        'vzsun': [xopt[1]],
        'mdisk_f': [xopt[2]],
        'disk_hz': [xopt[3]],
    }

    at.Table(xopt).write(tmp_filename, overwrite=True)

    return tmp_filename


def combine_output(all_filename, this_cache_path, elem_name):
    import glob

    cache_glob_pattr = str(this_cache_path / f'tmp-*{elem_name}*.csv')

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


def main(pool, overwrite=False):
    tree_K = 32  # MAGIC NUMBER: set heuristically in Objective-function.ipynb
    bootstrap_K = 128  # MAGIC NUMBER

    for data_name, d in datasets.items():
        # TODO: make seed configurable?
        rnd = np.random.default_rng(seed=42)

        # loop over all elements
        tasks = []
        cache_paths = []
        cache_filenames = []
        for elem_name in elem_names[data_name]:
            print(f"Running element: {elem_name}")

            # TODO: if galah in data_name, filter on flag??

            this_cache_path = cache_path / data_name
            this_cache_filename = (this_cache_path /
                                   f'optimize-results-{elem_name}.csv')
            if this_cache_filename.exists() and not overwrite:
                print(f"Cache file exists for {elem_name}: "
                      f"{this_cache_filename}")
                continue

            atexit.register(combine_output,
                            this_cache_filename,
                            this_cache_path,
                            elem_name)
            cache_paths.append(this_cache_path)
            cache_filenames.append(this_cache_filename)

            # print("Optimizing with full sample to initialize bootstraps...")
            # obj = TorusImagingObjective(d.t, d.c, elem_name, tree_K=tree_K)
            # full_sample_res = obj.minimize(method="nelder-mead",
            #                                options=dict(maxiter=1024))
            # if not full_sample_res.success:
            #     print(f"FAILED TO CONVERGE: optimize for full sample failed "
            #           f"for {elem_name}")
            #     continue

            # print(f"Finished optimizing full sample: {full_sample_res.x}")
            # x0 = full_sample_res.x
            x0 = np.array([20.8, 7.78, 1.1, 0.28])  # HACK: init from fiducial

            for k in range(bootstrap_K):
                idx = rnd.choice(len(d), len(d), replace=True)
                obj = TorusImagingObjective(d[idx], elem_name,
                                            tree_K=tree_K)
                tmp_filename = (this_cache_path /
                                f'tmp-optimize-results-{elem_name}-{k}.csv')
                tasks.append((k, obj, x0, tmp_filename))

        print("Done setting up bootstrap samples - running pool.map() on "
              f"{len(tasks)} tasks")

        for _ in pool.map(worker, tasks):
            pass

        for this_cache_filename, this_cache_path, elem_name in zip(
                cache_filenames,
                cache_paths,
                elem_names[data_name]):
            combine_output(this_cache_filename, this_cache_path, elem_name)

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
