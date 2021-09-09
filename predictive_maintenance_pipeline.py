import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
from sklearn.model_selection import train_test_split,
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os


# Leitura dos dados do diretório
suf = '.csv'
pref = './PdM_'
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

# Função para alterar os tipos dos dados dentro do dataframe
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

# Alterando estrutura dos dados dentro do dataframe
for col in dfs_cols:
  data[col] = data[col].apply(convertData)
  data[col] = data[col].astype('str')

data = data.dropna()
print(data.dtypes)

# Criando novas colunas relacionadas a quantidade de dias antes da falha

# #  Mais de 30 dias | Menos de 30 dias
class_over30 = "> 30 dias"
class_under30 = "< 30 dias"
data['30dias'] = class_over30
for index, row in data[data.failure != '0'].iterrows():
  data.loc[index-30*24:index, '30dias'] = class_under30


# # Menos de 15 dias | Mais de 15 dias
class_under15 = '< 15 dias'
class_over15 = '> 15 dias'
data["15dias"] = class_over15
for index, row in data[data.failure != '0'].iterrows():
  start = index-(15*24)
  data.loc[start:index, '15dias'] = class_under15


# # Menos de 7 dias | Mais de 7 dias
class_under7 = '< 7 dias'
class_over7 = '> 7 dias'
data["7dias"] = class_over7
for index, row in data[data.failure != '0'].iterrows():
  data.loc[index-7*24:index, '7dias'] = class_under7


# # Menos de 4 dias | Mais de 4 dias
class_under4 = '< 4 dias'
class_over4 = '> 4 dias'
data["4dias"] = class_over4
for index, row in data[data.failure != '0'].iterrows():
  data.loc[index-4*24:index, '4dias'] = class_under4


# # Menos de 2 dias | Mais de 2 dias
class_under2 = '< 2dias'
class_over2 = '> 2dias'
data["2dias"] = class_over2
for index, row in data[data.failure != '0'].iterrows():
  data.loc[index-2*24:index, '2dias'] = class_under2

# Realizando feature engineering

data_filtered = data
mean_val = 2
24//mean_val

features_cols = ['volt', 'vibration', 'rotate', 'pressure']
mean_suf24 = '_mean24h'
sd_suf24 = '_std24h'
mean_suf2 = '_mean12h'
sd_suf2 = '_std12h'
max_suf2 = '_max_12h'
min_suf2 = '_min_12h'

for id in data_filtered.machineID.unique():
  for col in features_cols:
    data_filtered.loc[data_filtered.machineID == id, f'{col}{mean_suf24}'] = data_filtered.loc[data_filtered.machineID == id, col].rolling(24).mean()
    data_filtered.loc[data_filtered.machineID == id, f'{col}{mean_suf2}'] = data_filtered.loc[data_filtered.machineID == id, col].rolling(int(24/mean_val)).mean()
    data_filtered.loc[data_filtered.machineID == id, f'{col}{max_suf2}'] = data_filtered.loc[data_filtered.machineID == id, col].rolling(int(24/mean_val)).max()
    data_filtered.loc[data_filtered.machineID == id, f'{col}{min_suf2}'] = data_filtered.loc[data_filtered.machineID == id, col].rolling(int(24/mean_val)).min()
    data_filtered.loc[data_filtered.machineID == id, 'RUL'] = data_filtered.loc[data_filtered.machineID == id].groupby((data_filtered.failure != data_filtered.failure.shift()).cumsum()).cumcount(ascending=False)
data_filtered = data_filtered.dropna()
data_filtered.loc['RUL'] = data_filtered['RUL'].astype('int')

# Separando os dados que serão utilizados em teste e validação de dados de simulação
sim_machines = data_filtered[data_filtered.machineID > 95]
data_filtered = data_filtered[data_filtered.machineID < 96]

# Separando os dados em duas classes somente por dado
data30 = data_filtered.drop(columns=['15dias', '7dias', '4dias', '2dias'])
data15 = data_filtered.loc[data_filtered['30dias']  == class_under30].drop( columns=['30dias', '7dias', '4dias', '2dias'])
data7 = data_filtered.loc[data_filtered['15dias']   == class_under15].drop( columns=['30dias', '15dias', '4dias', '2dias'])
data4 = data_filtered.loc[data_filtered['7dias']  == class_under7].drop(  columns=['30dias', '15dias', '7dias', '4dias'])
data2 = data_filtered.loc[data_filtered['4dias']  == class_under4].drop(columns=['30dias', '15dias', '7dias', '4dias'])

# Garantindo que a última coluna do meu dataframe é a minha variável de saída
Two_days_col = data2['2dias']
data2 = data2.drop(columns=['2dias'])
data2['2dias'] = Two_days_col
print(f"Balançeamento dos dados:\n{data2['2dias'].value_counts(normalize=True)}")


# Preparando os dados para treinamento do modelo (normalização, exclusão de colunas inúteis, etc)
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
  temp_data = dataset.drop(columns = ['datetime', 'errorID', 'failure', 'machineID', 'RUL'])
  temp_data = temp_data.to_numpy()
  scaler.fit(temp_data[:, :-1])
  temp_scaled_data = scaler.transform(temp_data[:, :-1])
  x_train, x_test, y_train, y_test = train_test_split(temp_scaled_data, temp_data[:, -1])
  le.fit(y_train)
  x_train_list.append(x_train)
  x_test_list.append(x_test)
  y_train_list.append(le.transform(y_train))
  y_test_list.append(le.transform(y_test))
  scaler_list.append(scaler)
  encoder_list.append(le)


# Construção do modelo / 
model_name = 'agosto20_3'
try:
    loaded_model = tf.keras.models.load_model(f'./{model_name}')
except: 
    model = Sequential()
    model.add(Dense(256, input_shape=(x_train_list[0].shape[1],)))
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        metrics=['accuracy'],
        loss='binary_crossentropy'
    )

    history = model.fit(
        x_train_list[0],
        y_train_list[0],
        batch_size=32,
        epochs=200,
        validation_data=(x_test_list[0], y_test_list[0])
    )

    # Salvar o modelo no diretório
    model.save(f'./{model_name}')


# Visualização dos resultados de treinamento (loss e accuracy)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Resultado do Treinamento do Modelo')
ax1.plot(history.history['loss'])
ax1.plot(history.history['val_loss'])
ax1.set_xlabel('Época')
ax1.set_ylabel('Loss')
ax1.legend(['Treino', 'Validação'], loc='upper right')

ax2.plot(history.history['accuracy'])
ax2.plot(history.history['val_accuracy'])
ax2.set_xlabel('Época')
ax2.set_ylabel('Precisão')
ax2.legend(['Treino', 'Validação'], loc='upper left')
plt.show()

# Realizando predição para os dados de teste
try:
    y_pred = loaded_model.predict(x_test_list[0])
except:
    y_pred = model.predict(x_test_list[0])

y_pred = np.where(y_pred > 0.5, 1, 0)
y_pred = encoder_list[0].inverse_transform(y_pred.reshape(y_pred.shape[0],))
y_test = encoder_list[0].inverse_transform(y_test_list[0])

conf_mat = confusion_matrix(y_test, y_pred)
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

# Métricas (Desempenho) do modelo
# Métricas:
corretas = conf_mat[0,0] + conf_mat[1,1]
erradas = conf_mat[0,1] + conf_mat[1,0]
accuracy = round(100*corretas/(erradas+corretas), 2)
recall_maior = round(100*conf_mat[1,1]/(conf_mat[1,1] + conf_mat[1,0]), 2)
recall_menor = round(100*conf_mat[0,0]/(conf_mat[0,0] + conf_mat[0,1]), 2)
print(f'Precisão: {accuracy}%')
print(f'Recall de Operação Estável: {recall_maior}%')
print(f'Recall de Iminência de Falha: {recall_menor}%')