import os
from glob import glob

from visualize import setup_axis

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:85% !important; }</style>"))

from evaluation_helper import EvaluationHelper
