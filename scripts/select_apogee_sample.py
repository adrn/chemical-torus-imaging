import os
import astropy.coordinates as coord
from astropy.io import fits
import astropy.table as at
import matplotlib as mpl
import matplotlib.pyplot  # noqa: needed for path stuff below?!
import numpy as np

coord.galactocentric_frame_defaults.set('v4.0')
allstar_filename = os.path.expanduser(
    '~/data/APOGEE_beta/allStarLite-r13-l33-58932beta.fits')


def main(overwrite=False):
    this_path = os.path.split(os.path.abspath(__file__))[0]
    data_path = os.path.abspath(os.path.join(this_path, '../data'))
    os.makedirs(data_path, exist_ok=True)

    parent_filename = os.path.join(data_path, 'apogee-parent-sample.fits')
    if os.path.exists(parent_filename) and not overwrite:
        print(f"Parent sample file already exists at {parent_filename}")
        return

    allstar = fits.getdata(allstar_filename)
    # astronn = fits.getdata('/Users/apricewhelan/data/APOGEE_beta/apogee_astroNN-r13-l33-58932beta.fits')

    aspcap_bitmask = np.sum(2**np.array([
        7,  # STAR_WARN
        23  # STAR_BAD
    ]))
    qual_mask = (
        (allstar['LOGG'] < 3.5) &
        (allstar['LOGG'] > 1) &
        (allstar['TEFF'] < 6500) &
        (allstar['TEFF'] > 3500) &
        (allstar['SNR'] > 40) &
        (allstar['FE_H'] > -3) &
        (allstar['MG_FE'] > -1) &
        ((allstar['ASPCAPFLAG'] & aspcap_bitmask) == 0)
    )
    stars = allstar[qual_mask]

    # save_cols = ['APOGEE_ID', 'dist', 'dist_error', 'dist_model_error',
    #              'nn_parallax', 'nn_parallax_error', 'nn_parallax_model_error',
    #              'fakemag', 'fakemag_error', 'weighted_dist',
    #              'weighted_dist_error', 'pmra', 'pmra_error', 'pmdec',
    #              'pmdec_error', 'phot_g_mean_mag', 'bp_rp', 'age',
    #              'age_linear_correct', 'age_lowess_correct', 'age_total_error',
    #              'age_model_error', 'source_id', 'jr', 'jr_err', 'Lz',
    #              'Lz_err', 'jz', 'jz_err', 'jr_Lz_corr', 'jr_jz_corr',
    #              'lz_jz_corr', 'omega_r', 'omega_r_err', 'omega_phi',
    #              'omega_phi_err', 'omega_z', 'omega_z_err', 'theta_r',
    #              'theta_r_err', 'theta_phi', 'theta_phi_err', 'theta_z',
    #              'theta_z_err', 'rl', 'rl_err', 'Energy', 'Energy_err',
    #              'EminusEc', 'EminusEc_err']
    # t = at.join(at.Table(stars),
    #             at.Table(astronn)[save_cols],
    #             keys='APOGEE_ID')

    # Remove stars targeted in known clusters
    mask_bits = {
        'APOGEE_TARGET1': np.array([9, 18, 24, 26]),
        'APOGEE_TARGET2': np.array([10, 18]),
        'APOGEE2_TARGET1': np.array([9, 18, 20, 21, 22, 23, 24, 26]),
        'APOGEE2_TARGET2': np.array([10]),
        'APOGEE2_TARGET3': np.array([5, 14, 15])
    }
    target_mask = np.ones(len(stars), dtype=bool)
    for name, bits in mask_bits.items():
        target_mask &= (stars[name] & np.sum(2**bits)) == 0

    # Select chemical "thin disk"
    mh_alpham_nodes = np.array([[0, -0.1],
                                [0.6, -0.03],
                                [0.5, 0.04],
                                [0.15, 0.04],
                                [-0.5, 0.13],
                                [-0.9, 0.13],
                                [-1., 0.07],
                                [0, -0.1]])
    mh_alpham_path = mpl.path.Path(mh_alpham_nodes[:-1])
    thin_disk_mask = mh_alpham_path.contains_points(
        np.stack((stars['M_H'],
                  stars['ALPHA_M'])).T)

    main_mask = target_mask & thin_disk_mask
    print(f"{main_mask.sum()} stars pass quality and targeting masks")
    at.Table(stars[main_mask]).write(parent_filename)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--overwrite', action='store_true',
                        dest='overwrite',
                        help=('Boolean flag to determine whether to overwrite '
                              'any existing generated data files.'))
    args = parser.parse_args()
    main(overwrite=args.overwrite)
