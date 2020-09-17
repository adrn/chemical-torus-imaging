
def elem_to_label(elem, dollar=True):
    num, den = elem.split('_')

    if num.lower() == 'alpha' and den.lower() == 'm':
        label = r'[\alpha / {\rm M}]'

    else:
        label = rf"[{{\rm {num.title()} }} / {{\rm {den.title()} }}]"

    if dollar:
        return '$' + label + '$'
    else:
        return label
