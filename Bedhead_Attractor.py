#=================================================================================
#-------------------------      Bedhead Attractor     ----------------------------
#=================================================================================

#-------------     X =  sin(X * Y / b) * Y + cos(a * X - Y)    -------------------
#-------------     Y =  sin(Y) / b + X                        --------------------

#---------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import pandas as pd
import sys
import datashader as ds
from datashader import transfer_functions as tf
from datashader.colors import Greys9, inferno, viridis
from datashader.utils import export_image
from functools import partial
from numba import jit
import numba
from colorcet import palette

#---------------------------------------------------------------------------------

background = "white"
img_map = partial(export_image, export_path="bedhead_maps", background=background)

n = 15000000

#---------------------------------------------------------------------------------

@jit
def trajectory(fn, a, b, x0=0, y0=0, n=n):

    x, y = np.zeros(n), np.zeros(n)
    x[0], y[0] = x0, y0

    for i in np.arange(n-1):
        x[i+1], y[i+1] = fn(a, b, x[i], y[i])

    return pd.DataFrame(dict(x=x, y=y))

@jit
def bedhead(a, b, x, y):

    return np.sin(x*y/b)*y + np.cos(a*x - y),   np.sin(y)/b + x

#---------------------------------------------------------------------------------

cmaps =  [palette[p][::-1] for p in ['kg', 'kr', 'kb', 'bjy', 'blues', 'bmy']]
cmaps += [inferno[::-1], viridis[::-1]]
cvs = ds.Canvas(plot_width = 800, plot_height = 800)
ds.transfer_functions.Image.border=0

#---------------------------------------------------------------------------------

# Parameter  :              a=xxx,   b=xxx
df = trajectory(bedhead,    1.37,    1.23,   1,   1)
#df = trajectory(bedhead,   -0.81,   -0.92,   1,   1)
#df = trajectory(bedhead,    1.67,    1.83,   1,   1)
#df = trajectory(bedhead,   -0.67,    0.33,   1,   1)

# Try to put a value in xxx.
#df = trajectory(bedhead,     xxx,     xxx,   1,   1)

agg = cvs.points(df, 'x', 'y')
img = tf.shade(agg, cmap = cmaps[4], how='linear', span = [0, n/60000])
img_map(img,"attractor")

#---------------------------------------------------------------------------------

# This program must be run in a Jupyter notebook.

