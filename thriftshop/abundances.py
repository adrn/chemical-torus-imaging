
def get_elem_names(tbl, min_frac=0.75):
    elem_names = ['FE_H'] + sorted([x for x in tbl.colnames
                                    if x.endswith('_FE')])
    f_good_meas = [((tbl[e] > -3) & (tbl[e] < 3)).sum() / len(tbl)
                   for e in elem_names]
    elem_names = [e for e, f in zip(elem_names, f_good_meas) if f > min_frac]
    elem_names.pop(elem_names.index('TIII_FE'))
    return elem_names


def elem_to_label(elem):
    num, den = elem.split('_')

    if num.lower() == 'alpha' and den.lower() == 'm':
        return r'$[\alpha / {\rm M}]$'

    else:
        return f'[{num.title()}/{den.title()}]'
