import pathlib
import astropy.table as at
import astropy.units as u
import numpy as np
from pyia import GaiaData

from .config import apogee_parent_filename, galah_parent_filename


class Dataset:

    def __init_subclass__(cls, **kwargs):
        if not hasattr(cls, '_radial_velocity_name'):
            cls._radial_velocity_name = 'radial_velocity'
        super().__init_subclass__(**kwargs)

    def __init__(self, filename_or_tbl):
        if (isinstance(filename_or_tbl, str)
                or isinstance(filename_or_tbl, pathlib.Path)):
            self.t = at.QTable.read(filename_or_tbl)
        else:
            self.t = at.QTable(filename_or_tbl)
        self.t = self._init_mask()

        # Abundance ratios should be all caps:
        for col in self.t.colnames:
            if col.upper().endswith('_FE') or col.upper().startswith('FE_'):
                self.t.rename_column(col, col.upper())

        self.g = GaiaData(self.t)

        # Use Gaia RV if not defined at dataset subclass level
        rv_name = self._radial_velocity_name

        rv = u.Quantity(self.t[rv_name])
        if rv.unit.is_equivalent(u.one):
            rv = rv * u.km/u.s
        self.c = self.g.get_skycoord(radial_velocity=rv)

    def _init_mask(self):
        # TODO: implement on subclasses
        return self.t

    @property
    def elem_ratios(self):
        if not hasattr(self, '_elem_ratios'):
            self._elem_ratios = ['FE_H'] + sorted([x for x in self.t.colnames
                                                   if x.endswith('_FE') and
                                                   not x.startswith('E_') and
                                                   not x.startswith('FLAG_')])
        return self._elem_ratios

    @property
    def elem_names(self):
        if not hasattr(self, '_elem_names'):
            elem_list = ([x.split('_')[0] for x in self.elem_ratios] +
                         [x.split('_')[1] for x in self.elem_ratios])
            elem_list.pop(elem_list.index('H'))
            self._elem_names = set(elem_list)
        return self._elem_names

    def get_elem_ratio(self, elem1, elem2=None):
        if elem2 is None:
            try:
                elem1, elem2 = elem1.split('_')
            except Exception:
                raise RuntimeError("If passing a single elem ratio string, "
                                   "it must have the form ELEM_ELEM, not "
                                   f"{elem1}")

        elem1 = str(elem1).upper()
        elem2 = str(elem2).upper()

        if elem2 == 'H':
            i1 = self.elem_ratios.index(elem1 + 'FE')
            i2 = self.elem_ratios.index('FE_H')
            return (self.t[self.elem_ratios[i1]] -
                    self.t[self.elem_ratios[i2]])

        else:
            i1 = self.elem_ratios.index(elem1 + '_FE')
            i2 = self.elem_ratios.index(elem2 + '_FE')
            return (self.t[self.elem_ratios[i1]] -
                    self.t[self.elem_ratios[i2]])

    def filter(self, filters):
        mask = np.ones(len(self.t), dtype=bool)
        for k, (x1, x2) in filters.items():
            if x1 is None and x2 is None:
                raise ValueError("Doh")

            arr = u.Quantity(self.t[k]).value

            if x1 is None:
                mask &= arr < x2

            elif x2 is None:
                mask &= arr >= x1

            else:
                mask &= (arr >= x1) & (arr < x2)

        return self[mask]

    def __getitem__(self, slc):
        if isinstance(slc, int):
            slc = slice(slc, slc+1)
        return self.__class__(self.t[slc])


class APOGEEDataset(Dataset):
    _radial_velocity_name = 'VHELIO_AVG'

    def _init_mask(self):
        aspcap_bitmask = np.sum(2 ** np.array([
            # 7,  # STAR_WARN
            23  # STAR_BAD
        ]))
        quality_mask = (
            (self.t['SNR'] > 20) &
            ((self.t['ASPCAPFLAG'] & aspcap_bitmask) == 0)
        )

        # Remove stars targeted in known clusters or dwarf galaxies:
        mask_bits = {
            'APOGEE_TARGET1': np.array([9, 18, 24, 26]),
            'APOGEE_TARGET2': np.array([10, 18]),
            'APOGEE2_TARGET1': np.array([9, 18, 20, 21, 22, 23, 24, 26]),
            'APOGEE2_TARGET2': np.array([10]),
            'APOGEE2_TARGET3': np.array([5, 14, 15])
        }
        target_mask = np.ones(len(self.t), dtype=bool)
        for name, bits in mask_bits.items():
            target_mask &= (self.t[name] & np.sum(2**bits)) == 0

        return self.t[quality_mask & target_mask]


class GALAHDataset(Dataset):
    _radial_velocity_name = 'rv_synt'

    def _init_mask(self):
        quality_mask = (self.t['flag_cannon'] == 0)

        # Remove stars targeted in known clusters or dwarf galaxies:
        # TODO: how to do this for GALAH??

        return self.t[quality_mask]


apogee = APOGEEDataset(apogee_parent_filename)
galah = GALAHDataset(galah_parent_filename)

# datasets = {
#     'apogee-rgb-loalpha': apogee.filter({'LOGG': (1, 3.5),
#                                          'TEFF': (3500, 6500),
#                                          'FE_H': (-3, 1)},
#                                         low_alpha=True),
#     'apogee-rgb-hialpha': apogee.filter({'LOGG': (1, 3.5),
#                                          'TEFF': (3500, 6500),
#                                          'FE_H': (-3, 1)},
#                                         low_alpha=False)
# }

# datasets = {
#     'apogee-rgb-loalpha': APOGEEDataset(filter={'logg': (1, 3.5),
#                                                 'teff': (3500, 6500)),
#     'apogee-rgb-hialpha': APOGEEDataset(),
#     'apogee-ms-loalpha': APOGEEDataset(),
#     'apogee-ms-hialpha': APOGEEDataset(),
#     'galah-rgb-loalpha': GALAHDataset(),
#     'galah-rgb-hialpha': GALAHDataset(),
#     'galah-ms-loalpha': GALAHDataset(),
#     'galah-ms-hialpha': GALAHDataset(),
# }