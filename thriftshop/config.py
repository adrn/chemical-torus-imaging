import pathlib
import astropy.units as u
import numpy as np

# Eilers et al. 2019:
vcirc = 229. * u.km/u.s
rsun = 8.122 * u.kpc
fiducial_mdisk = 6.52551e10 * u.Msun

# Assumes that the package isn't actually installed, but built in place
pkg_path = pathlib.Path(__file__).resolve().parent.parent
cache_path = pkg_path / 'cache'
plot_path = pkg_path / 'plots'
fig_path = pkg_path / 'tex/figures'
for p in [plot_path, fig_path, cache_path]:
    p.mkdir(exist_ok=True)

# Plotting config stuff (limits, units, etc.)
zlim = 1.5  # kpc
vlim = 80.  # km/s
Rlim = (7, 9.5)  # kpc
plot_config = {
    "zunit": u.kpc,
    "vunit": u.km/u.s,
    "Junit": u.kpc * u.km/u.s,
    "zlim": zlim,
    "vlim": vlim,
    "vticks": np.arange(-vlim, vlim+1e-3, vlim//2),
    "zticks": np.round(np.arange(-zlim, zlim+1e-3, zlim/3), 2),
    "vminorticks": np.arange(-vlim, vlim+1e-3, vlim//4),
    "zminorticks": np.round(np.arange(-zlim, zlim+1e-3, zlim/6), 2),
    "Rlim": Rlim,
    "Rticks": np.arange(*Rlim, 1),
    "Rminorticks": np.arange(*Rlim, 0.5)
}
