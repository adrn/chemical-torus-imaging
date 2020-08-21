import pathlib
import astropy.units as u

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
