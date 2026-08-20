from golf_hole import GolfHole
from golf_hole_plot import drawHole
import os

sample_hole = os.path.join('assets', 'sample_par4.kml')
golf_hole = GolfHole(sample_hole)


drawHole(golf_hole)
