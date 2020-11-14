
# coding: utf-8

# In[ ]:


from pandas import DataFrame
from pandas import concat
import pandas as pd
import numpy as np
import seaborn as sns
import os,inspect
import sys
from sklearn import preprocessing


# In[ ]:


path_file = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
path_file=os.path.join(path_file,'datos_reales')


# In[ ]:


#read file from csv or excel
def readFile(file, sheet=None):
    if file.endswith('.xlsx') or file.endswith('.xls'):
        if sheet == None:
            raise Exception('sheet name should be defined')
        else:
            return pd.read_excel(open(os.path.join(path_file,file), 'rb'), sheet_name=sheet)
    elif file.endswith('.csv'):
        return pd.read_csv(os.path.join(path_file,file))
    else:
        raise Exception('Types would be 0 or 1')


# In[ ]:


#ruta: ruta del origen xlsx
#sheet_name: nombre de hoja
def loadData(ruta, sheet_name):
    data = readFile(ruta,sheet_name).set_index("year")
    return data


# In[ ]:


#data: dataFrame imput
#n_in: cantidad de datos historicos para inferir un resultado
#n_out: numero de resultados
#dropnan: true borra las columnas vacias, else los considera
def series_to_supervised(data, n_in=1, n_out=1, dropnan=False):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# In[ ]:


#dataFrame:particiona funcion series_to_supervised()
#n_train: pariciona el dataFrame para el training, el restante se considera validacion("test")
def splitData(dataFrame,n_train, n_in, n_out, var):
    #numero de datos
    longData = len(dataFrame.index)
    x_train_años = (longData * n_train)//100
    # split into train and test sets
    train = dataFrame.values[:x_train_años, :]
    test = dataFrame.values[x_train_años:, :]
    # split into input and outputs
    x_train, y_train = train[:, :var*n_in], train[:, var*n_in:]
    x_val, y_val = test[:, :var*n_in], test[:, var*n_in:]
    # reshape input to be 3D [samples, timesteps, features]
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1],1))
    x_val = x_val.reshape((x_val.shape[0], x_val.shape[1],1 ))
    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    return x_train, y_train, x_val, y_val


# In[ ]:

#firts call indicatorPaisPerAnno
#_dataFrame = indicatorPaisPerAnno
#normalization data and return data frame
def normalizeData(_dataFrame, _transpose=False):
    std_scale = preprocessing.StandardScaler().fit(_dataFrame)
    data_norm = std_scale.transform(_dataFrame)

    data_norm_col = pd.DataFrame(data_norm, index=_dataFrame.index, columns=_dataFrame.columns) 
    _dataFrame.update(data_norm_col)
    if _transpose:
        return _dataFrame.T
    return _dataFrame

#scale data berween -1 to 1
def scale(_dataFrame,a=-1, b=1):
    maxVal = _dataFrame.iloc[:,:].max()
    minVal = _dataFrame.iloc[:,:].min()
    data = (b-a)*((_dataFrame.iloc[:,:] - minVal)/(maxVal-minVal))+a
    _dataFrame.iloc[:,:]=data
    return _dataFrame
#los n_in primeros no tienen historicos y se considera NAN
def data(fileName, n_in=4, n_out=1, n_train=85,sheet_name="hoja1"):
    data=loadData(fileName, sheet_name)
    #data = normalizeData(data)
    data = scale(data)
    dataFrame=series_to_supervised(data, n_in, n_out)
    dataFrame = dataFrame.iloc[n_in:,:]
    var = len(data.columns)
    return splitData(dataFrame,n_train, n_in, n_out, var), var

