# Standard library
import atexit
import os

# Third-party
from astropy.utils import iers
iers.conf.auto_download = False
import astropy.table as at

# This project
from thriftshop.config import cache_path
from thriftshop.data import load_apogee_sample
from thriftshop.potentials import potentials
from thriftshop.actions_multiproc import action_worker


def this_action_worker(task):
    (i1, i2), t, potential_name, cache_path = task

    cache_filename = os.path.join(cache_path,
                                  f'tmp-{potential_name}-{i1}-{i2}.fits')

    print(f"Worker {i1} processing {len(t)} stars, saving to {cache_filename}")

    data = action_worker([(i1, i2), t, potential_name])

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

    # Register exit command for each potential
    all_filenames = {}
    for potential_name in potentials:
        all_filenames[potential_name] = os.path.join(
            cache_path, f'aaf-{potential_name}.fits')
        atexit.register(combine_output,
                        all_filenames[potential_name],
                        cache_path,
                        potential_name)

    t, c = load_apogee_sample(data_filename)

    # Execute worker on batches:
    for potential_name, potential in potentials.items():
        if os.path.exists(all_filenames[potential_name]):
            print(f"Actions exist for {potential_name} at "
                  f"{all_filenames[potential_name]}")
            continue

        tasks = batch_tasks(n_batches=max(1, pool.size),
                            arr=t,
                            args=(potential_name, cache_path))

        sub_filenames = []
        for sub_filename in pool.map(this_action_worker, tasks):
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
