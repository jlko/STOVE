"""Contains common imports and jupyter notebook setups.

Execute with `%run jupyter_start.py` at the top of a notebook.
"""

import os
import sys
from glob import glob

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:85% !important; }</style>"))
