import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, SimpleRNN, GRU, LSTM, Dense, Flatten, GlobalMaxPool1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

# https://suzuki-navi.hatenablog.com/entry/2020/09/22/073756
# https://suzuki-navi.hatenablog.com/entry/2020/08/16/235911

#=============================================================
# Example of custom optimizer
#=============================================================



