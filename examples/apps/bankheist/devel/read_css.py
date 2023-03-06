import cssutils
from pprint import pprint

dct = {}
sheet = cssutils.parseString("../sprites/cars/police/isosprites.css")


# for rule in sheet:
#     selector = rule.selectorText
#     styles = rule.style.cssText
#     dct[selector] = styles


pprint(dct)
