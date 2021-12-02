# -*- coding: utf-8 -*-
"""02/12/2021 - Pipeline de desenvolvimento da rede neural final"""
"""Autor: Ricardo Bohadana Martins"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
# Commented out IPython magic to ensure Python compatibility.
from ann_visualizer.visualize import ann_viz
from google.colab import drive
from keras.layers import Dense
from keras.models import Sequential
from matplotlib import rc
from matplotlib.colors import ListedColormap
from matplotlib.dates import (ConciseDateFormatter, DateFormatter, DayLocator,
                              HourLocator)
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from tensorflow import keras

drive.mount('/content/drive')
# %matplotlib inline

"""# Leitura dos dados"""

pref = 'drive/MyDrive/Colab Data/PdM_'
suf = '.csv'
telemetry = pd.read_csv(f'{pref}telemetry{suf}', parse_dates=['datetime'])
errors = pd.read_csv(f'{pref}errors{suf}', parse_dates=['datetime'])
failures = pd.read_csv(f'{pref}failures{suf}', parse_dates=['datetime'])
machines = pd.read_csv(f'{pref}machines{suf}')
maint = pd.read_csv(f'{pref}maint{suf}', parse_dates=['datetime'])

dfs = [errors, failures]
dfs_cols = ['errorID', 'failure']

data = telemetry
for df in dfs:
  data = pd.merge(data, df, how='outer', on=['datetime', 'machineID'])

def convertData(val):
  """
  Convert de column value to str value
  - NA/NaN/inf returns 0
  - else returns last character
  """
  if str(val) == 'nan':
    new_val = '0'
  else:
    new_val = str(val)[-1]
  return str(new_val)

for col in dfs_cols:
  data[col] = data[col].apply(convertData)
  data[col] = data[col].astype('str')

data = data.dropna()
data.dtypes

# for index, row in data.iterrows():
#   if row.failure == "0":
#     if row.errorID != "0":
#       data.loc[index, 'failure'] = row.errorID

# Retirando as máquinas que não apresentaram falhas no período
ids = []
for id in data.machineID.unique():
  temp_data = data[data.machineID == id]
  if len(temp_data[temp_data.failure != "0"]) < 1:
    data = data[data.machineID !=  id]
    ids.append(id)
print(f'IDs de máquinas retiradas: {ids}')



"""# Divisão dos dados
> Divisão de todos os dados em datasets balanceados com 2 classes cada um em relação à quantidade de dias para falha. (30, 15, 7, 3.5, 2, 1) 


"""

# Dividir em duas classes
# #  Mais de 30 dias
# #  Menos de 30 dias
class_over30 = "> 30 dias"
class_under30 = "< 30 dias"
data['30dias'] = class_over30
for index, row in data[data.failure != '0'].iterrows():
  data.loc[index-30*24:index, '30dias'] = class_under30

# Dividir em mais duas classes
# # Menos de 15 dias
# # Mais de 15 dias
class_under15 = '< 15 dias'
class_over15 = '> 15 dias'
data["15dias"] = class_over15
for index, row in data[data.failure != '0'].iterrows():
  start = index-(15*24)
  data.loc[start:index, '15dias'] = class_under15

# Dividir em mais duas classes
# # Menos de 7 dias
# # Mais de 7 dias
class_under7 = '< 7 dias'
class_over7 = '> 7 dias'
data["7dias"] = class_over7
for index, row in data[data.failure != '0'].iterrows():
  data.loc[index-7*24:index, '7dias'] = class_under7

# Dividir em mais duas classes
# # Menos de 3.5 dias
# # Mais de 3.5 dias
class_under4 = '< 4 dias'
class_over4 = '> 4 dias'
data["4dias"] = class_over4
for index, row in data[data.failure != '0'].iterrows():
  data.loc[index-4*24:index, '4dias'] = class_under4

# Dividir em mais duas classes
# # Menos de 2 dias
# # Mais de 2 dias
class_under2 = '< 2dias'
class_over2 = '> 2dias'
data["2dias"] = class_over2
for index, row in data[data.failure != '0'].iterrows():
  data.loc[index-2*24:index, '2dias'] = class_under2

# # Separando os dados em duas classes somente por dado

# data30 = data.drop(columns=['15dias', '7dias', '4dias', '2dias'])
# data15 = data.loc[data['30dias']  == class_under30].drop( columns=['30dias', '7dias', '4dias', '2dias'])
# data7 = data.loc[data['15dias']   == class_under15].drop( columns=['30dias', '15dias', '4dias', '2dias'])
# data4 = data.loc[data['7dias']  == class_under7].drop(  columns=['30dias', '15dias', '7dias', '2dias'])
# data2 = data.loc[data['4dias']  == class_under4].drop(columns=['30dias', '15dias', '7dias', '4dias'])

# data2['2dias'].value_counts(normalize=True)

# filter = (data['2dias'] == '> 2dias')
# data_filtered = data[filter]
# for col in ['vibration', 'rotate', 'volt', 'pressure']:
#   Q1 = data_filtered[col].quantile(0.15)
#   Q3 = data_filtered[col].quantile(0.85)
#   IQR = Q3 - Q1
#   print(col)
#   print(Q1-IQR)
#   print(Q3+IQR)

#   filter_col = (data_filtered[col] >= Q1 - IQR) & (data_filtered[col] <= Q3 + IQR)
#   data_filtered = data_filtered[filter_col]

# data_filtered = data_filtered.append(data[data['2dias'] == '< 2dias'])
# data_filtered = data_filtered.sort_values(by='datetime')
data_filtered = data

"""# Criando novos features


> Volt, Vibration, Pressure and Rotation **mean last 24h**

> Volt, Vibration, Pressure and Rotation **mean last 2d**

> Volt, Vibration, Pressure and Rotation **max last 2d**

> Volt, Vibration, Pressure and Rotation **min last 2d**

"""

# errorCondition = data.errorID == '0'
# data[errorCondition].failure.value_counts()
# data = data[errorCondition]
# data
mean_val = 0.5
24//mean_val

# executa em 3s para 876445 linhas
features_cols = ['volt', 'vibration', 'rotate', 'pressure']
mean_suf24 = '_mean24h'
sd_suf24 = '_std24h'
mean_suf2 = '_mean12h'
sd_suf2 = '_std12h'
max_suf2 = '_max_6h'
min_suf2 = '_min_6h'

for id in data_filtered.machineID.unique():
  for col in features_cols:
    data_filtered.loc[data_filtered.machineID == id, f'{col}_ewma24h'] = data_filtered.loc[data_filtered.machineID == id, col].ewm(span=24).mean()
    data_filtered.loc[data_filtered.machineID == id, f'{col}{mean_suf24}'] = data_filtered.loc[data_filtered.machineID == id, col].rolling(24).mean()
    # data_filtered.loc[data_filtered.machineID == id, f'{col}{sd_suf24}'] = data_filtered.loc[data_filtered.machineID == id, col].rolling(24).std()
    # data_filtered.loc[data_filtered.machineID == id, f'{col}{mean_suf2}'] = data_filtered.loc[data_filtered.machineID == id, col].rolling(int(24/mean_val)).mean()
    # data_filtered.loc[data_filtered.machineID == id, f'{col}{sd_suf2}'] = data_filtered.loc[data_filtered.machineID == id, col].rolling(int(24/mean_val)).std()
    # data_filtered.loc[data_filtered.machineID == id, f'{col}{max_suf24}'] = data_filtered.loc[data_filtered.machineID == id, col].rolling(24).max()
    data_filtered.loc[data_filtered.machineID == id, f'{col}{max_suf2}'] = data_filtered.loc[data_filtered.machineID == id, col].rolling(int(6)).max()
    data_filtered.loc[data_filtered.machineID == id, f'{col}{min_suf2}'] = data_filtered.loc[data_filtered.machineID == id, col].rolling(int(6)).min()
    # data_filtered.loc[data_filtered.machineID == id, 'RUL'] = data_filtered.loc[data_filtered.machineID == id].groupby((data_filtered.failure != data_filtered.failure.shift()).cumsum()).cumcount(ascending=False)
# data_filtered.drop(columns = features_cols)
data_filtered = data_filtered.dropna()
# data_filtered['RUL'] = data_filtered['RUL'].astype('int')

# data['Problem'] = 0

# for index, row in data.iterrows():
#   if row.failure != '0':
#     for i in range(0,49):
#       data.loc[index-i, 'Problem'] = 1

sim_machines = data_filtered[data_filtered.machineID > 95]
data_filtered = data_filtered[data_filtered.machineID < 96]

# plt.figure(figsize=(20,10))
# # sns.lineplot(data=data[data.machineID == 42][:481], x='datetime', y='volt_mean24h')
# # sns.lineplot(data=data[data.machineID == 42][:481], x='datetime', y='volt_mean2d')
# # sns.lineplot(data=data[data.machineID == 42][:481], x='datetime', y='volt')
# sns.scatterplot(data=data[data.machineID == 42][:1500], x='datetime',y='volt_mean2d', hue='failure', sizes=(250,1500))

# Separando os dados em duas classes somente por dado
data30 = data_filtered.drop(columns=['15dias', '7dias', '4dias', '2dias'])
data15 = data_filtered.loc[data_filtered['30dias']  == class_under30].drop( columns=['30dias', '7dias', '4dias', '2dias'])
data7 = data_filtered.loc[data_filtered['15dias']   == class_under15].drop( columns=['30dias', '15dias', '4dias', '2dias'])
data4 = data_filtered.loc[data_filtered['7dias']  == class_under7].drop(  columns=['30dias', '15dias', '7dias', '4dias'])
data2 = data_filtered.loc[data_filtered['4dias']  == class_under4].drop(columns=['30dias', '15dias', '7dias', '4dias'])

data_filtered.to_pickle('/content/data_filtered')



"""# Formatação de gráficos para o formato científico"""

# Commented out IPython magic to ensure Python compatibility.
# %%bash
# git clone https://github.com/garrettj403/SciencePlots.git
# cd SciencePlots
# pip install -e .

data_filtered.to_pickle('data')

!pip install latex

plt.style.reload_library()
plt.style.use(['science', 'notebook', 'grid', 'no-latex'])
# plt.rcParams.update({
#     "font.family": "Times New Roman",   # specify font family here
#     "font.serif": ["Times"],  # specify font here
#     "font.size":10})          # specify font size here

"""# Pré processamento dos dados"""

data2.columns

Two_days_col = data2['2dias']
data2 = data2.drop(columns=['2dias'])
data2['2dias'] = Two_days_col

data2['2dias'].value_counts(normalize=True)

# Leu os dados da base?
# data_list = [all_data['data2']]
# Executou os scripts?
data_list = [data2]

x_train_list = []
x_test_list = []
y_train_list = []
y_test_list = []
scaler_list = []
encoder_list = []
for dataset in data_list:
  scaler = MinMaxScaler()
  le = LabelEncoder()
  # temp_data = dataset.drop(columns = ['datetime', 'errorID', 'failure', 'machineID', 'RUL']).to_numpy()
  temp_data = dataset.drop(columns = ['datetime', 'errorID', 'failure', 'machineID'] + features_cols)
  temp_data = temp_data.to_numpy()
  scaler.fit(temp_data[:, :-1])
  temp_scaled_data = scaler.transform(temp_data[:, :-1])
  x_train, x_test, y_train, y_test = train_test_split(temp_scaled_data, temp_data[:, -1])
  print(x_train.shape)
  le.fit(y_train)
  x_train_list.append(x_train)
  x_test_list.append(x_test)
  y_train_list.append(le.transform(y_train))
  y_test_list.append(le.transform(y_test))
  scaler_list.append(scaler)
  encoder_list.append(le)

data2['2dias'].value_counts(normalize=True)

scaler_list

x_train_list[0]

"""# Validação Cruzada"""

import time

from sklearn.model_selection import KFold

num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True)
splits = list(kfold.split(x_train_list[0]))
train_indices = splits[0][0]

class ComputationalEffortHistory(keras.callbacks.Callback):
  def __init__(self, model_number):
    self.logs = []
    self.model_number = model_number

  def on_epoch_begin(self, epoch, logs={}):
    self.epoch_time_start = time.time()

  def on_epoch_end(self, epoch, logs={}):
    self.logs.append(time.time() - self.epoch_time_start)

cross_epochs = 30
cross_batch_size = 30
print(f'Epochs: {cross_epochs}\nBatch Size: {cross_batch_size}')

models_1esc = []
models_2esc = []
computational_effort = []
nodes1 = list(range(4, 17, 4))
nodes2 = list(range(2, 9, 2))
for i, item in enumerate(nodes1):
  
  models_1esc.append(
      Sequential(
          [
            Dense(item, activation='sigmoid', input_shape=(x_train_list[0].shape[1], )),
            Dense(1, activation='sigmoid')
          ]
      )
  )
  models_2esc.append(
      Sequential(
          [
            Dense(item, activation='sigmoid', input_shape=(x_train_list[0].shape[1], )),
            Dense(nodes2[i], activation='sigmoid'),
            Dense(1, activation='sigmoid')
          ]
      )
  )

models = models_1esc + models_2esc

history_list = []

for index, model in enumerate(models):

  timing_callback = ComputationalEffortHistory(index)
  computational_effort.append(timing_callback)

  model.compile(
      optimizer='adam',
      metrics=['accuracy'],
      loss='binary_crossentropy',
  )

  history_list.append(
      model.fit(
        x_train_list[0][train_indices],
        y_train_list[0][train_indices],
        batch_size=cross_batch_size,
        epochs=cross_epochs,
        validation_data=(x_test_list[0], y_test_list[0]),
        callbacks=[timing_callback]
        )
  )

for model in models:
  model.summary()

legend = []
for i, model in enumerate(models):
  if i < 4:
    legend.append(str([16, nodes1[i], 1]))
  else:
    legend.append(str([16, nodes1[i-4], nodes2[i-4], 1]))

print(legend)

fig, ax = plt.subplots(1,1, figsize=(12,8))
ax.set_title('Precisão do modelo por Época')
ax.set_ylabel('Precisão')
ax.set_xlabel('Épocas')
colors = ['blue', 'orange', 'red', 'black', 'grey', 'green', 'cyan', 'purple']

# ax.legend(['', 'val'], loc='upper left')
for index, history in enumerate(history_list):
  # ax.plot(history.history['val_loss'])
  ax.plot(history.epoch[5:], history.history['val_accuracy'][5:], color=colors[index])

# ax.legend([str([16,12,1]), str([16,12,6,1])])
ax.legend(legend)
plt.plot()
fig.savefig('/content/cross_validation_accuracy_zoom.png')

fig, ax = plt.subplots(1,1, figsize=(12,8))
ax.set_title('Loss (Erro) do modelo por época')
ax.set_ylabel('Loss (Erro)')
ax.set_xlabel('Épocas')
colors = ['blue', 'orange', 'red', 'black', 'grey', 'green', 'cyan', 'purple']


# ax.legend(['', 'val'], loc='upper left')
for index, history in enumerate(history_list):
  # ax.plot(history.history['val_loss'])
  ax.plot(history.epoch[5:], history.history['val_loss'][5:], color=colors[index])

# ax.legend([str([16,12,1]), str([16,12,6,1])])
ax.legend(legend)
plt.plot()
fig.savefig('/content/cross_validation_loss_zoom.png')

fig, ax = plt.subplots(1,1, figsize=(12,8))
ax.set_title('Tempo de Treinamento por Época')
ax.set_ylabel('Esforço Computacional (s)')
ax.set_xlabel('Épocas')
colors = ['blue', 'orange', 'red', 'black', 'grey', 'green', 'cyan', 'purple']

# ax.legend(['', 'val'], loc='upper left')
for index, history in enumerate(history_list):
  # ax.plot(history.history['val_loss'])
  ax.plot(history.epoch, computational_effort[index].logs, color=colors[index])

# ax.legend([str([16,12,1]), str([16,12,6,1])])
ax.legend(legend)
plt.plot()
fig.savefig('/content/cross_validation_computational_effort.png')

"""# Treinamento dos melhores modelos"""

final_models = [
  
  Sequential(
      [
        Dense(16, activation='sigmoid', input_shape=(x_train_list[0].shape[1], )),
        Dense(1, activation='sigmoid')
      ]
  ),
  Sequential(
      [
        Dense(12, activation='sigmoid', input_shape=(x_train_list[0].shape[1], )),
        Dense(6, activation='sigmoid'),
        Dense(1, activation='sigmoid')
      ]
  )
]
final_history = []
final_computational_effort = []

for i, model in enumerate(final_models):

  final_timing_callback = ComputationalEffortHistory(i)
  final_computational_effort.append(final_timing_callback)

  model.compile(
    optimizer='adam',
    metrics=['accuracy'],
    loss='binary_crossentropy'
  )

  final_history.append(
    model.fit(
      x_train_list[0],
      y_train_list[0],
      batch_size=32,
      epochs=200,
      validation_data=(x_test_list[0], y_test_list[0]),
      callbacks=[final_timing_callback]
    )
  )

colors = ['black', 'cyan']
text_colors = ['white', 'black']

"""# Visualização do resultado dos treinamentos"""

fig, ax = plt.subplots(1,1, figsize=(12,8))
ax.set_title('Precisão do modelo por Época')
ax.set_ylabel('Precisão')
ax.set_xlabel('Épocas')


# ax.legend(['', 'val'], loc='upper left')
for index, history in enumerate(final_history):
  # ax.plot(history.history['val_loss'])
  text = str(
      round(history.history['val_accuracy'][-1], 3)
  )
  ax.plot(history.epoch[30:], history.history['val_accuracy'][30:], color=colors[index])
  ax.text(75, 0.8919 - index/500, text, backgroundcolor=colors[index], color=text_colors[index])

ax.legend([str([16,8,4,1]), str([16,12,6,1])])
# ax.legend(legend)
plt.plot()
fig.savefig('/content/cross_validation_accuracy_zoom_final.png')

fig, ax = plt.subplots(1,1, figsize=(12,8))
ax.set_title('Precisão do modelo por Época')
ax.set_ylabel('Precisão')
ax.set_xlabel('Épocas')

# ax.legend(['', 'val'], loc='upper left')
for index, history in enumerate(final_history):
  # ax.plot(history.history['val_loss'])
  text = str(
      round(history.history['accuracy'][-1], 3)
  )
  ax.plot(history.epoch[30:], history.history['val_accuracy'][30:], color=colors[index])
  ax.plot(history.epoch[30:], history.history['accuracy'][30:], '--', color=colors[index], lw=0.5)
  ax.text(75, 0.8919 - index/500, text, backgroundcolor=colors[index], color=text_colors[index])

ax.legend([str([16,8,4,1]), str([16,12,6,1])])
# ax.legend(legend)
plt.plot()
fig.savefig('/content/cross_validation_accuracy_zoom_final_train.png')

fig, ax = plt.subplots(1,1, figsize=(12,8))
ax.set_title('Loss (Erro) do modelo por época')
ax.set_ylabel('Loss (Erro)')
ax.set_xlabel('Épocas')


# ax.legend(['', 'val'], loc='upper left')
for index, history in enumerate(final_history):
  # ax.plot(history.history['val_loss'])
  text = str(round(history.history['val_loss'][-1], 5))
  ax.plot(history.epoch[5:], history.history['val_loss'][5:], color=colors[index])
  ax.text(135, 0.378-index/250, text, backgroundcolor=colors[index], color=text_colors[index])

ax.legend([str([16,8,4,1]), str([16,12,6,1])])
# ax.legend(legend)
plt.plot()
fig.savefig('/content/cross_validation_loss_zoom_final.png')


fig, ax = plt.subplots(1,1, figsize=(12,8))
ax.set_title('Loss (Erro) do modelo por época')
ax.set_ylabel('Loss (Erro)')
ax.set_xlabel('Épocas')


# ax.legend(['', 'val'], loc='upper left')
for index, history in enumerate(final_history):
  # ax.plot(history.history['val_loss'])
  text = str(round(history.history['loss'][-1], 5))
  ax.plot(history.epoch[5:], history.history['loss'][5:], color=colors[index])
  ax.text(135, 0.378-index/250, text, backgroundcolor=colors[index], color=text_colors[index])

ax.legend([str([16,8,4,1]), str([16,12,6,1])])
# ax.legend(legend)
plt.plot()
fig.savefig('/content/cross_validation_loss_zoom_final_train.png')

def seconds_to_minutes_seconds(seconds: int):
  minutes = seconds//60
  sec = seconds % 60

  return int(minutes), int(sec)


fig, ax = plt.subplots(1,1, figsize=(12,8))
ax.set_title('Tempo de Treinamento por Época')
ax.set_ylabel('Esforço Computacional (s)')
ax.set_xlabel('Épocas')

# ax.legend(['', 'val'], loc='upper left')
for index, history in enumerate(final_history):
  # ax.plot(history.history['val_loss'])
  min, secs = seconds_to_minutes_seconds(sum(final_computational_effort[index].logs))
  text = f'{min}m{secs}s'
  ax.plot(history.epoch, final_computational_effort[index].logs, color=colors[index])
  ax.text(130, 9-index/8, text, backgroundcolor=colors[index], color=text_colors[index])

ax.legend([str([16,8,4,1]), str([16,12,6,1])])
# ax.legend(legend)
plt.plot()
fig.savefig('/content/cross_validation_computational_effort_final.png')

"""# Gráfico da variável de saída"""

fDate = DateFormatter('%d/%m')
pseudo_data = data.query('machineID == 33')[610:800]
value_encoder = {'< 2dias': 0, '> 2dias': 1}
pseudo_data['2dias_binary'] = pseudo_data['2dias'].apply(lambda row: value_encoder[row])

fig, ax = plt.subplots(3,1, figsize=(12,8), sharex=True)
# fig.suptitle('Comportamento da Variável de Saída')
ax[0].set_ylabel('Ponto de Falha')
ax[1].set_ylabel('Tempo até a Falha')
ax[2].set_ylabel('Saída do Modelo')
ax[0].xaxis.set_major_formatter(fDate)
ax[1].xaxis.set_major_formatter(fDate)
ax[2].xaxis.set_major_formatter(fDate)
ax[2].set_xlabel('Data')

ax[0].plot(pseudo_data.datetime, pseudo_data.failure)
ax[1].plot(pseudo_data.datetime, pseudo_data['2dias'])
ax[2].plot(pseudo_data.datetime, pseudo_data['2dias_binary'])

fig.savefig('/content/output_sample.png')



"""# Persistência dos modelos treinados"""

names=['02122021_16_16_1', '02122021_16_12_6_1']
for index, model in enumerate(final_models):
  model.save(f'/content/drive/MyDrive/Colab Data/Modelos/{names[index]}')

"""# Teste do modelo em tempo real"""

# y definition a confusion matrix C is such that C[i,j] is equal to the number of observations known to be in group i and predicted to be in group j.

# y_pred = loaded_model.predict(x_test_list[0])
y_pred = final_models[0].predict(x_test_list[0])
y_pred = np.where(y_pred > 0.5, 1, 0)
y_pred = encoder_list[0].inverse_transform(y_pred.reshape(y_pred.shape[0],))
y_test = encoder_list[0].inverse_transform(y_test_list[0])

conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12,10))
plt.title("Matriz de Confusão")
sns.heatmap(
    conf_mat,
    annot=True,
    fmt='d',
    yticklabels=['Falha em menos de 2 dias', 'Operação Estável'],
    xticklabels=['Previsto - Falha em menos de 2 dias', 'Previsto - Operação Estável'],
    cmap="Blues",
    cbar=False,
    robust=True,  
    square=True,
    linewidths=20,
    linecolor='White'
)
# plt.savefig('/content/drive/MyDrive/Colab Data/ConfusionMatrix_azuredb.png', dpi=300)



"""# Simulação temporal aplicada da máquina 100"""

default_sim_machines = sim_machines
filter = (sim_machines['2dias'] == '> 2dias')
sim_machines_filtered = sim_machines[filter]


for col in ['vibration', 'rotate', 'volt', 'pressure']:
  Q1 = sim_machines_filtered[col].quantile(0.3)
  Q3 = sim_machines_filtered[col].quantile(0.7)
  IQR = Q3 - Q1

  filter_col = (sim_machines_filtered[col] >= Q1 - IQR) & (sim_machines_filtered[col] <= Q3 + IQR)
  sim_machines_filtered = sim_machines_filtered[filter_col]

sim_machines = sim_machines_filtered.append(sim_machines[sim_machines['2dias'] == '< 2dias'])
sim_machines = sim_machines.sort_values(by='datetime')

# data = all_data['all_data']
features_cols = ['volt', 'vibration', 'rotate', 'pressure']
sim_machines_2dias = sim_machines['2dias']
sim_machines = sim_machines.drop(columns=['2dias'])
sim_machines['2dias'] = sim_machines_2dias
machine_100 = sim_machines[sim_machines.machineID == 100]
# machine_100.columns
remove_cols = ['datetime', 'errorID', 'failure', 'machineID', '30dias', '15dias', '7dias', '4dias']

temp_data = machine_100.drop(columns = remove_cols+features_cols).dropna().to_numpy()

# temp_data = machine_100.drop(columns = remove_cols).dropna().to_numpy()

scaler = scaler_list[0]
le = LabelEncoder()
# pré-processamento


# Dados de entrada
# scaler.fit(temp_data[:, :-1])
temp_scaled_data = scaler.transform(temp_data[:, :-1])
x_test = temp_scaled_data

# Dados de saída
le.fit(temp_data[:, -1])
y_test = le.transform(temp_data[:, -1])
x_test

import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list('mycmap', ['green', 'yellow', 'orange', 'red'])

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

# date_format = DateFormatter('%d/%m - %Hh')
# prediction = model.predict(x_test)
prediction = final_models[0].predict(x_test)
prediction = np.where(prediction > 0.3, 1, 0)
# prediction = np.where(prediction > 0.5, 1, prediction)
# prediction = np.where(prediction < 0.2, 0, np.where(prediction <= 0.5, 0.5, prediction))
machine_100['prediction'] = prediction
machine_100['prediction_perc'] = (1 - machine_100['prediction'].ewm(span=24).mean())
# graph_start = 0
# graph_end = len(machine_100)
graph_start = -395
graph_end = -250
machine_100_50 = machine_100[graph_start:graph_end]

fig, ax = plt.subplots(9, 1, figsize=(30, 18), sharex='col', sharey='row')
for i in range(0, 9):
  # ax[i].xaxis.set_major_formatter(date_format)
  ax[i].xaxis.set_minor_locator(HourLocator(interval=20))
  ax[i].xaxis.set_major_locator(DayLocator())
  ax[i].xaxis.set_minor_formatter(ConciseDateFormatter(ax[i].xaxis.get_minor_locator()))
  ax[i].xaxis.set_major_formatter(ConciseDateFormatter(ax[i].xaxis.get_major_locator()))



# Primeira Linha
ax[0].plot(machine_100_50.datetime, machine_100_50.rotate_ewma24h, color='blue')
ax[0].plot(machine_100_50.datetime, machine_100_50.rotate_mean24h, color='black')

ax[0].set_title('Rotação')


# Segunda Linha
ax[1].plot(machine_100_50.datetime, machine_100_50.vibration_ewma24h, color='blue')
ax[1].plot(machine_100_50.datetime, machine_100_50.vibration_mean24h, color='black')
ax[1].set_title('Vibração')


# Terceira Linha
ax[2].plot(machine_100_50.datetime, machine_100_50.pressure_ewma24h, color='blue')
ax[2].plot(machine_100_50.datetime, machine_100_50.pressure_mean24h, color='black')
ax[2].set_title('Pressão')


# Quarta Linha
ax[3].plot(machine_100_50.datetime, machine_100_50.volt_ewma24h, color='blue')
ax[3].plot(machine_100_50.datetime, machine_100_50.volt_mean24h, color='black')
ax[3].set_title('Tensão')


# Quinta Linha
ax[4].plot(machine_100_50.datetime, machine_100_50.errorID, color='black')
ax[4].plot(machine_100_50.datetime, machine_100_50.errorID, color='black')
ax[4].set_title('Erro')

# Sexta Linha
ax[5].plot(machine_100_50.datetime, machine_100_50.failure)
ax[5].set_title('Falha')

# Sétima Linha
ax[6].plot(machine_100_50.datetime, le.transform(machine_100_50['2dias']))
ax[6].set_title('Valor Real')

# Sétima Linha
ax[7].plot(machine_100_50.datetime, prediction[graph_start:graph_end], color='black')
ax[7].set_title('Predição')

c = mcolors.ColorConverter().to_rgb
rvb = make_colormap(
    [c('green'), c('red')])
colors = machine_100_50.prediction_perc
# Sétima Linha

ax[8].scatter(machine_100_50.datetime,  machine_100_50.prediction_perc, c=colors, cmap=rvb)
ax[8].plot(machine_100_50.datetime, [0.65]*len(machine_100_50.datetime), '--', lw=1.5, color='#EED202')
ax[8].set_title('Probabilidade de Falha')

for i in range(0, 4):
  ax[i].legend(['MME (24)', 'MMS (24h)'])


fig.tight_layout(pad=2.0)
fig.savefig('final_simulation_zoom.png')

# date_format = DateFormatter('%d/%m - %Hh')
# prediction = model.predict(x_test)
prediction = final_models[0].predict(x_test)
prediction = np.where(prediction > 0.3, 1, 0)
# prediction = np.where(prediction > 0.5, 1, prediction)
# prediction = np.where(prediction < 0.2, 0, np.where(prediction <= 0.5, 0.5, prediction))
machine_100['prediction'] = prediction
machine_100['prediction_perc'] = (1 - machine_100['prediction'].ewm(span=24).mean())
graph_start = 0
graph_end = len(machine_100)
machine_100_50 = machine_100[graph_start:graph_end]

fig, ax = plt.subplots(9, 1, figsize=(30, 18), sharex='col', sharey='row')
for i in range(0, 9):
  ax[i].xaxis.set_major_formatter(date_format)
  # ax[i].xaxis.set_minor_locator(HourLocator(interval=20))
  # ax[i].xaxis.set_major_locator(DayLocator())
  # ax[i].xaxis.set_minor_formatter(ConciseDateFormatter(ax[i].xaxis.get_minor_locator()))
  # ax[i].xaxis.set_major_formatter(ConciseDateFormatter(ax[i].xaxis.get_major_locator()))



# Primeira Linha
ax[0].plot(machine_100_50.datetime, machine_100_50.rotate_ewma24h, color='blue')
ax[0].plot(machine_100_50.datetime, machine_100_50.rotate_mean24h, color='black')

ax[0].set_title('Rotação')


# Segunda Linha
ax[1].plot(machine_100_50.datetime, machine_100_50.vibration_ewma24h, color='blue')
ax[1].plot(machine_100_50.datetime, machine_100_50.vibration_mean24h, color='black')
ax[1].set_title('Vibração')


# Terceira Linha
ax[2].plot(machine_100_50.datetime, machine_100_50.pressure_ewma24h, color='blue')
ax[2].plot(machine_100_50.datetime, machine_100_50.pressure_mean24h, color='black')
ax[2].set_title('Pressão')


# Quarta Linha
ax[3].plot(machine_100_50.datetime, machine_100_50.volt_ewma24h, color='blue')
ax[3].plot(machine_100_50.datetime, machine_100_50.volt_mean24h, color='black')
ax[3].set_title('Tensão')


# Quinta Linha
ax[4].plot(machine_100_50.datetime, machine_100_50.errorID, color='black')
ax[4].plot(machine_100_50.datetime, machine_100_50.errorID, color='black')
ax[4].set_title('Erro')

# Sexta Linha
ax[5].plot(machine_100_50.datetime, machine_100_50.failure)
ax[5].set_title('Falha')

# Sétima Linha
ax[6].plot(machine_100_50.datetime, le.transform(machine_100_50['2dias']))
ax[6].set_title('Valor Real')

# Sétima Linha
ax[7].plot(machine_100_50.datetime, prediction[graph_start:graph_end], color='black')
ax[7].set_title('Predição')

c = mcolors.ColorConverter().to_rgb
rvb = make_colormap(
    [c('green'), c('red')])
colors = machine_100_50.prediction_perc
# Sétima Linha

ax[8].scatter(machine_100_50.datetime,  machine_100_50.prediction_perc, c=colors, cmap=rvb)
ax[8].plot(machine_100_50.datetime, [0.65]*len(machine_100_50.datetime), '--', lw=1.5, color='#EED202')
ax[8].set_title('Probabilidade de Falha')

for i in range(0, 4):
  ax[i].legend(['MME (24)', 'MMS (24h)'])


fig.tight_layout(pad=2.0)
fig.savefig('final_simulation.png')

# plt.style.use('dark_background')
# prediction = model.predict(x_test)
prediction = final_models[0].predict(x_test)
prediction = np.where(prediction > 0.32, 1, 0)
# prediction = np.where(prediction > 0.5, 1, prediction)
# prediction = np.where(prediction < 0.2, 0, np.where(prediction <= 0.5, 0.5, prediction))
machine_100['prediction'] = prediction
machine_100['prediction_perc'] = (1 - machine_100['prediction'].ewm(span=48).mean())
graph_start = len(machine_100)-450
graph_end = len(machine_100)-200
machine_100_50 = machine_100[graph_start:graph_end]

date_format = DateFormatter('%d/%m')

fig, ax = plt.subplots(5, 1, figsize=(10, 10), sharex='col')
for i in range(0, 5):
  ax[i].xaxis.set_major_formatter(date_format)
  # ax[i, 1].xaxis.set_major_formatter(date_format)

# Primeira Linha
ax[0].plot(machine_100_50.datetime, machine_100_50.rotate_ewma24h, color='black')
ax[0].set_ylabel('Rotação')

# Segunda Linha
ax[1].plot(machine_100_50.datetime, machine_100_50.rotate_mean24h, color='black')
ax[1].set_ylabel('Média (12 horas)')

# Terceira Linha
ax[2].plot(machine_100_50.datetime, machine_100_50.failure, color='black')
ax[2].set_ylabel('Falha')

# Quarta Linha
ax[3].plot(machine_100_50.datetime, le.transform(machine_100_50['2dias']), color='black')
ax[3].set_ylabel('Saída Esperado')

# Quinta Linha
ax[4].plot(machine_100_50.datetime, prediction[graph_start:graph_end], color='black')
ax[4].set_ylabel('Predição')

fig.savefig('/content/final_evaluation_zoom.png')
# alterar a característica da saída para 2 variáveis e escrever sobre esse 
# plano na parte de metodologia





