
# coding: utf-8

import sys
#pathToGAM = 'global-attribution-mapping/'
pathToGAM = '/home/brian/src/global-attribution-mapping/'
sys.path.insert(0, pathToGAM)
from gam import gam


local_attribution_path = 'bias_deeplift_attributions-50-50-basline-1k-samples.csv'
g = gam.GAM(attributions_path = local_attribution_path, k=3)
g.generate()

g.plot(num_features=5)
g.subpopulation_sizes
