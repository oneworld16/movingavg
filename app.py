import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from math import sqrt
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pickle

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from tqdm import tqdm_notebook
from itertools import product

import statsmodels.api as sm
import pandas as pd
import streamlit as st


st.title('Gold price trend Prediction')

#dataset
st.subheader('Data')
df = pd.read_csv('FINAL_USO.csv')
st.write(df.tail(5))
#describe the data
st.subheader('Data Description from 2011-2018')
st.write(df.describe())

#visualization
st.subheader('Adjusted Closing Price vs Time Chart')
fig = plt.figure(figsize=(8,6))
plt.plot(df['Adj Close'])
st.pyplot(fig)

st.subheader("Chart with 100MA")
ma100 = df['Adj Close'].rolling(100).mean()
fig = plt.figure(figsize=(6,4))
plt.plot(df['Adj Close'])
plt.plot(ma100)
st.pyplot(fig)

st.subheader("Chart with 200MA")
ma200 = df['Adj Close'].rolling(200).mean()
fig = plt.figure(figsize=(6,4))
plt.plot(df['Adj Close'])
plt.plot(ma200)
st.pyplot(fig)

st.subheader("Chart with 100MA & 200MA")
fig = plt.figure(figsize=(6,4))
plt.plot(df['Adj Close'])
plt.plot(ma100)
plt.plot(ma200)
st.pyplot(fig)





#prediction with linear Regression

