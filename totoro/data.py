import astropy.coordinates as coord
import astropy.table as at
import astropy.units as u


def load_apogee_sample(filename):
    t = at.Table.read(filename)
    t = t[(t['GAIA_PARALLAX'] > 0.666) &
          ((t['GAIA_PARALLAX'] / t['GAIA_PARALLAX_ERROR']) > 5)]

    c = coord.SkyCoord(ra=t['RA'] * u.deg,
                       dec=t['DEC'] * u.deg,
                       distance=1000 / t['GAIA_PARALLAX'] * u.pc,
                       pm_ra_cosdec=t['GAIA_PMRA'] * u.mas/u.yr,
                       pm_dec=t['GAIA_PMDEC'] * u.mas/u.yr,
                       radial_velocity=t['VHELIO_AVG'] * u.km/u.s)

    return t, c
