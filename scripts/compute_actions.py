# Third-party
from astropy.utils import iers
iers.conf.auto_download = False
import gala.dynamics as gd

# This project
from thriftshop.config import cache_path, galcen_frame
from thriftshop.data import load_apogee_sample
from thriftshop.potentials import potentials, galpy_potentials
from thriftshop.galpy_helpers import StaeckelAAFComputer


def worker(task):
    data_filename, aaf_computer, potential_name, aaf_filename = task

    # Read APOGEE sample and do parallax & plx s/n cut:
    t, c = load_apogee_sample(data_filename)
    galcen = c.transform_to(galcen_frame)
    w0 = gd.PhaseSpacePosition(galcen.data)

    if aaf_filename.exists():
        print(f"Actions exist for {potential_name} at {aaf_filename}")
        return

    aaf = aaf_computer.get_aaf(w0, float(potential_name),
                               galpy_potentials[potential_name])
    aaf.write(aaf_filename)


def main(pool, data_filename):

    # Set up action solver:
    aaf_computer = StaeckelAAFComputer()

    tasks = []
    for potential_name in potentials:
        tasks.append((data_filename,
                      aaf_computer,
                      potential_name,
                      cache_path / f'aaf-{potential_name}.fits'))

    for _ in pool.map(worker, tasks):
        pass


if __name__ == '__main__':
    import sys
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser()

    parser.add_argument("--data", dest="data_filename", required=True,
                        type=str, help="the source data file")

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
