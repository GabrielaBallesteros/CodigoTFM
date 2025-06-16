# -*- coding: utf-8 -*-
"""
@author: Gabriela Ballesteros Gómez 
Código TFM
"""
###############################################################################
# preparamos los datos 

import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
datos= pd.read_excel("datosTFMG.xlsx")

# añadimos la variable categórica para saber si una póliza se ha dado de baja al año siguiente o no
datos["baja_prox_ano"] = (datos["ano_baja"] == datos["ano"] + 1).astype(int)

# añadimos el tiempo en cartera:
datos['tiempo_cartera']=datos['ano']-datos['aalt']

# transformación log(1+x) para el número de beneficiarios
datos['log_beneficiarios'] = np.log1p(datos['numBeneficiarios'])

# elimino edades<0 o >100
datos.loc[datos['edad'] < 0, 'edad'] = 0
datos.loc[datos['edad'] > 100, 'edad'] = 100


# importamos librerías
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.over_sampling import SMOTE

np.random.seed(13)
random.seed(13)
tf.random.set_seed(13)

###############################################################################
"""
MODELO 1: MLP + weights
"""
# -----------------------------
# 1. Cargar y preparar datos
# -----------------------------
# Separar X e y. En X elimino las variables que no voy a utilizar para predecir
X = datos.drop(columns=['id_pol', 'log_beneficiarios', 'anac', 'ano_baja', 'id_asg']) 

# -----------------------------
# 2. División train / test
# -----------------------------
df_train1 = X[X['ano'] < 2018]

# Validación: año 2018
df_val1 =X[X['ano'] == 2018]

# Test: año 2019
df_test1 = X[X['ano'] == 2019]

# Variables independientes y dependientes
feature_cols = [col for col in X.columns if col not in ('baja_prox_ano', 'ano')] #tampoco utilizaré ano, pero la necesitaba para separar entre train y test

X_train1 = df_train1[feature_cols]
y_train1 = df_train1['baja_prox_ano']

X_val1 = df_val1[feature_cols]
y_val1 = df_val1['baja_prox_ano']

X_test1 = df_test1[feature_cols]
y_test1 = df_test1['baja_prox_ano']

# -----------------------------
# 3. Preprocesado: normalizar y codificar
# -----------------------------
# Columnas categóricas y numéricas
cat_cols = ['producto', 'suc']
num_cols = [col for col in X_train1.columns if col not in cat_cols]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
])

# Ajustar en training, transformar todo
X_train_prep1 = preprocessor.fit_transform(X_train1)
X_val_prep1 = preprocessor.transform(X_val1)
X_test_prep1 = preprocessor.transform(X_test1)


# -----------------------------
# 4. Calcular class weights
# -----------------------------
weights1 = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train1),
    y=y_train1
)
class_weights1 = {0: weights1[0], 1: weights1[1]}
print(class_weights1)

# -----------------------------
# 5. Construcción de la red
# -----------------------------
n1,n2, n3 =(64, 64, 32) # número de neuronas
a1, a2, a3 = ('leaky_relu','elu','leaky_relu') # funciones de activación
d1, d2, d3 = (0, 0, 0) # dropout

model1 = Sequential()
model1.add(Dense(n1, input_dim=X_train_prep1.shape[1], activation=a1,  kernel_regularizer=l2(0.001)))
model1.add(Dropout(d1))
model1.add(Dense(n2, activation=a2))
model1.add(Dropout(d2))
model1.add(Dense(n3, activation=a3))
model1.add(Dropout(d3))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# -----------------------------
# 6. Entrenamiento
# -----------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


history1 = model1.fit(
        X_train_prep1, y_train1,
        validation_data=(X_val_prep1, y_val1),
        epochs=30,
        batch_size=32,
        class_weight=class_weights1,
        callbacks=[early_stop],
        verbose=1
        )


# -----------------------------
# 7. Evaluación
# -----------------------------
loss, acc = model1.evaluate(X_test_prep1, y_test1)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {acc:.4f}")

# Predicciones
y_proba1 = model1.predict(X_test_prep1).ravel()
y_pred1 = (y_proba1 >= 0.53).astype(int)

# Matriz de confusión
cm1=confusion_matrix(y_test1, y_pred1)
# Visualmente 
ax=sns.heatmap(cm1, annot=True, fmt="d", cmap="BuPu", xticklabels=[0,1], yticklabels=[0,1])
ax.set_xlabel("Predicción")
ax.set_ylabel("Real")
ax.set_title("Modelo 1: Matriz de confusión")

# Métricas
print(classification_report(y_test1, y_pred1, digits=3))
print("AUC-ROC:", roc_auc_score(y_test1, y_proba1))

# -----------------------------
# 8. Gráficas
# -----------------------------
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history1.history['loss'], label='Train Loss', color='hotpink')
plt.plot(history1.history['val_loss'], label='Val Loss', color='cadetblue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Evolución de la pérdida')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history1.history['accuracy'], label='Train Accuracy', color='hotpink')
plt.plot(history1.history['val_accuracy'], label='Val Accuracy', color='cadetblue')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Evolución de la accuracy')
plt.legend()

plt.tight_layout()
plt.show()

###############################################################################
"""
MODELO 2: MLP + SMOTE
"""
# -----------------------------
# 2. División train / test
# -----------------------------
# misma que en el modelo 1

# -----------------------------
# 3. Preprocesado: normalizar y codificar
# -----------------------------
# mismas que antes

# Transformar los datos
X_train_prep2 = preprocessor.fit_transform(X_train1)
X_val_prep2 = preprocessor.transform(X_val1)
X_test_prep2 = preprocessor.transform(X_test1)

# -----------------------------
# 4. Aplicar SMOTE-ENN
# -----------------------------

smote = SMOTE(random_state=42)
X_train_bal2, y_train_bal2 = smote.fit_resample(X_train_prep2, y_train1)

print(f"Clases tras SMOTE: {np.bincount(y_train_bal2)}")

# -----------------------------
# 5. Modelo
# -----------------------------
model2 = Sequential()
model2.add(Dense(64, activation='relu', input_shape=(X_train_bal2.shape[1],)))
model2.add(BatchNormalization())
model2.add(Dropout(0.5))
model2.add(Dense(32, activation='relu'))
model2.add(BatchNormalization())
model2.add(Dropout(0.3))
model2.add(Dense(32, activation='leaky_relu'))
model2.add(Dropout(0.1))
model2.add(Dense(1, activation='sigmoid'))

model2.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# -----------------------------
# 6. Entrenamiento
# -----------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history2 = model2.fit(
    X_train_bal2, y_train_bal2,
    validation_data=(X_val_prep2, y_val1),
    epochs=30,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)


# -----------------------------
# 7. Evaluación
# -----------------------------
loss, acc = model2.evaluate(X_test_prep2, y_test1)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {acc:.4f}")

# Predicción de probabilidades 
y_proba2 = model2.predict(X_test_prep2).ravel()
y_pred2 = (y_proba2 >= 0.53).astype(int)

# Matriz de confusión
cm2=confusion_matrix(y_test1, y_pred2)
# Visualmente:
ax=sns.heatmap(cm2, annot=True, fmt="d",  cmap="BuPu", xticklabels=[0,1], yticklabels=[0,1])
ax.set_xlabel("Predicción")
ax.set_ylabel("Real")
ax.set_title("Modelo 2: Matriz de confusión")

#Métricas
print(classification_report(y_test1, y_pred2, digits=3))
print("AUC-ROC:", roc_auc_score(y_test1, y_proba2))


# -----------------------------
# 8. Gráficas
# -----------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history2.history['loss'], label='Train Loss', color='hotpink')
plt.plot(history2.history['val_loss'], label='Val Loss', color='cadetblue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Evolución de la pérdida')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history2.history['accuracy'], label='Train Accuracy', color='hotpink')
plt.plot(history2.history['val_accuracy'], label='Val Accuracy', color='cadetblue')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Evolución de la Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

###############################################################################
"""
MODELO 3: MLP + SMOTE-ENN
"""
# -----------------------------
# 2. División train / test
# -----------------------------
df_train3 = X[X['ano'] <= 2018]

# Test: año 2019
df_test3 = X[X['ano'] == 2019]


X_train3 = df_train3[feature_cols]
y_train3 = df_train3['baja_prox_ano']

X_test3 = df_test3[feature_cols]
y_test3 = df_test3['baja_prox_ano']

# -----------------------------
# 3. Preprocesado: normalizar y codificar
# -----------------------------
# Transformar los datos
X_train_prep3= preprocessor.fit_transform(X_train3)
X_test_prep3 = preprocessor.transform(X_test3)


# -----------------------------
# 4. Aplicar SMOTE-ENN
# -----------------------------
smote_enn = SMOTEENN(random_state=42)
X_train_bal3, y_train_bal3 = smote_enn.fit_resample(X_train_prep3, y_train3)

print(f"Clases tras SMOTE-ENN: {np.bincount(y_train_bal3)}")

# -----------------------------
# 5. Red Neuronal
# -----------------------------

np.random.seed(13)
random.seed(13)
tf.random.set_seed(13)
d1, d2 = (0.5, 0.1) #dropout 

model3 = Sequential()
model3.add(Dense(256, activation='relu', input_shape=(X_train_bal3.shape[1],)))
model3.add(Dropout(d1))
model3.add(Dense(128, activation='elu'))
model3.add(BatchNormalization())
model3.add(Dropout(d2))
model3.add(Dense(1, activation='sigmoid'))
model3.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# -----------------------------
# 6. Entrenamiento
# -----------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history3 = model3.fit(
    X_train_bal3, y_train_bal3,
    validation_split=0.2,
    epochs=30,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# 7. Evaluación
# -----------------------------
loss3, acc3 = model3.evaluate(X_test_prep3, y_test3)
print(f"Test Loss: {loss3:.4f}")
print(f"Test Accuracy: {acc3:.4f}")

# Predicción de probabilidades
y_proba3 = model3.predict(X_test_prep3).ravel()
y_pred3 = (y_proba3 >= 0.51).astype(int)

# Matriz de confusión
cm3=confusion_matrix(y_test3, y_pred3)
# Visualmente:
ax=sns.heatmap(cm3, annot=True, fmt="d",  cmap="BuPu", xticklabels=[0,1], yticklabels=[0,1])
ax.set_xlabel("Predicción")
ax.set_ylabel("Real")
ax.set_title("Modelo 3: Matriz de confusión")

# Métricas
print(classification_report(y_test3, y_pred3, digits=3))
print("AUC-ROC:", roc_auc_score(y_test3, y_proba3))

# -----------------------------
# 8. Gráficas
# -----------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history3.history['loss'], label='Train Loss', color='hotpink')
plt.plot(history3.history['val_loss'], label='Val Loss', color='cadetblue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Evolución de la pérdida')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history3.history['accuracy'], label='Train Accuracy', color='hotpink')
plt.plot(history3.history['val_accuracy'], label='Val Accuracy', color='cadetblue')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Evolución de la Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

###############################################################################
"""
MODELO 4: MLP + SMOTE-Tomek
"""
# 1., 2. y 3. igual que el Modelo 3
# ----------------------------
# 4. Aplicar SMOTE-Tomek
# ----------------------------
smt = SMOTETomek(random_state=42)
X_resampled4, y_resampled4 = smt.fit_resample(X_train_prep3, y_train3)

print("Antes del balanceo:", np.bincount(y_train3))
print("Después del balanceo:", np.bincount(y_resampled4))

# -----------------------------
# 5. Red Neuronal
# -----------------------------
n1, n2, n3 = (32, 32, 16)  #número de neuronas 

model4 = Sequential()
model4.add(Dense(n1, input_dim=X_train_prep3.shape[1], activation='elu'))
model4.add(BatchNormalization())
model4.add(Dropout(0.3))
model4.add(Dense(n2, activation='elu'))
model4.add(BatchNormalization())
model4.add(Dropout(0.1))
model4.add(Dense(n3, activation='elu'))
model4.add(BatchNormalization())
model4.add(Dropout(0.1))
model4.add(Dense(1, activation='sigmoid'))
model4.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# -----------------------------
# 6. Entrenamiento
# -----------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

np.random.seed(13)
random.seed(13)
tf.random.set_seed(13)

history4 = model4.fit(
        X_resampled4, y_resampled4,
        validation_split=0.2,
        epochs=30,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
      )

# -----------------------------
# 7. Evaluación
# -----------------------------
loss4, acc4 = model4.evaluate(X_test_prep3, y_test3)
print(f"Test Loss: {loss4:.4f}")
print(f"Test Accuracy: {acc4:.4f}")

# Predicción de probabilidades y ajuste del umbral (opcional)
y_proba4 = model4.predict(X_test_prep3)#.ravel()
y_pred4 = (y_proba4 >= 0.4).astype(int)


# Matriz de confusión
cm4=confusion_matrix(y_test3, y_pred4)
# Visualmente:
ax=sns.heatmap(cm4, annot=True, fmt="d",  cmap="BuPu", xticklabels=[0,1], yticklabels=[0,1])
ax.set_xlabel("Predicción")
ax.set_ylabel("Real")
ax.set_title("Modelo 4: Matriz de confusión")

#  Métricas
print(classification_report(y_test3, y_pred4, digits=3))
print("AUC-ROC:", roc_auc_score(y_test3, y_proba4))

# -----------------------------
# 8. Gráficas
# -----------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history4.history['loss'], label='Train Loss', color='hotpink')
plt.plot(history4.history['val_loss'], label='Val Loss', color='cadetblue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Evolución de la pérdida')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history4.history['accuracy'], label='Train Accuracy', color='hotpink')
plt.plot(history4.history['val_accuracy'], label='Val Accuracy', color='cadetblue')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Evolución de la Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

###############################################################################
"""
MODELO 5: LSTM 
"""    
# -----------------------------
# 3. Preprocesado: normalizar y codificar
# -----------------------------
datos.drop(columns=['baja_prox_ano', 'id_pol', 'ano_baja' , 'log_beneficiarios', 'id_asg', 'anac'])

# Define columnas
cat_cols = ['producto', 'suc']
num_cols = [col for col in datos.columns if col not in cat_cols + ['id_pol', 'ano', 'baja_prox_ano', 'id_asg', 'ano_baja', 'log_beneficiarios', 'anac']]
all_features = cat_cols + num_cols

# Preprocesador
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
])


# Fit-transform sobre todos los datos
X_transformed = preprocessor.fit_transform(datos)
feature_names = (
    preprocessor.named_transformers_['num'].get_feature_names_out(num_cols).tolist() +
    preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols).tolist()
)

# Combinar features procesados con las columnas id_asg, ano y respuesta
datos_features = pd.DataFrame(X_transformed, columns=feature_names, index=datos.index)
datos_processed = pd.concat([datos[['id_asg', 'ano', 'baja_prox_ano']], datos_features], axis=1)

# --------------------------
# 4. Convertir en secuencia
# --------------------------
# Rellenar años faltantes con 0s
X_seq5 = []
y_seq5 = []
años = sorted(datos_processed['ano'].unique())
n_features = datos_features.shape[1]

for pid, group in datos_processed.groupby('id_asg'):
    # Agrupar por año si hay duplicados por póliza y año
    group = group.groupby('ano').mean(numeric_only=True).reindex(años, fill_value=0)

    # Asegurar que todas las columnas necesarias estén presentes
    missing_cols = set(feature_names) - set(group.columns)
    for col in missing_cols:
        group[col] = 0  # completar con ceros si faltan columnas

    # Reordenar columnas para que estén en el mismo orden
    group = group[feature_names]

    features = group.values
    label = datos_processed.loc[(datos_processed['id_asg'] == pid) & (datos_processed['ano'] == años[-1]), 'baja_prox_ano']
    label = label.values[0] if len(label) > 0 else 0  # Si no hay valor para el último año, asumo 0

    X_seq5.append(features)
    y_seq5.append(label)

X_seq5 = np.array(X_seq5)
y_seq5 = np.array(y_seq5)

print("Shape final de X:", X_seq5.shape)
print("Shape final de y:", y_seq5.shape)

# --------------------------
# 5. Train/test split
# --------------------------
X_train5, X_test5, y_train5, y_test5 = train_test_split(X_seq5, y_seq5, test_size=0.2, random_state=42)

print("Shape de X_train:", X_train5.shape)
print("Shape de y_train:", y_train5.shape)
print("Shape de X_test:", X_test5.shape)
print("Shape de y_test:", y_test5.shape)


class_weights5 = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train5),
    y=y_train5
)
class_weights5 = dict(enumerate(class_weights5))

# -----------------------------
# 6. Red Neuronal
# -----------------------------
n1,n2=(128, 256) # número de neuronas 
d1, d2= (0.1, 0.3) # dropout 

timesteps5 = X_seq5.shape[1]      # = 5 (años)
features5 = X_seq5.shape[2]       # = 67 (features por año)


model5 = Sequential()
model5.add(LSTM(n1, input_shape=(timesteps5, features5), return_sequences=True)) 
model5.add(Dropout(d1))
model5.add(LSTM(n2, return_sequences=False))
model5.add(Dropout(d2))
model5.add(Dense(1, activation='sigmoid'))

model5.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# -----------------------------
# 7. Entrenamiento
# -----------------------------
np.random.seed(13)
random.seed(13)
tf.random.set_seed(13)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history5 = model5.fit(
        X_train5, y_train5,
        validation_split=0.2,
        epochs=30,
        batch_size=64,
        class_weight=class_weights5,
        callbacks=[early_stop],
        verbose=1
    )


# -----------------------------
# 8. Evaluación
# -----------------------------
loss5, acc5 = model5.evaluate(X_test5, y_test5)
print(f"Test Loss: {loss5:.4f}")
print(f"Test Accuracy: {acc5:.4f}")

# Predicción de probabilidades
y_proba5 = model5.predict(X_test5).ravel()
y_pred5 = (y_proba5 >= 0.62 ).astype(int)

# Matriz de confusión
cm5=confusion_matrix(y_test5, y_pred5)
# Visualmente:
ax=sns.heatmap(cm5, annot=True, fmt="d",  cmap="BuPu", xticklabels=[0,1], yticklabels=[0,1])
ax.set_xlabel("Predicción")
ax.set_ylabel("Real")
ax.set_title("Modelo 5: Matriz de confusión")

#Métricas
print(classification_report(y_test5, y_pred5, digits=3))
print("AUC-ROC:", roc_auc_score(y_test5, y_proba5))

# -----------------------------
# 9. Gráficas
# -----------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history5.history['loss'], label='Train Loss', color='hotpink')
plt.plot(history5.history['val_loss'], label='Val Loss', color='cadetblue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Evolución de la pérdida')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history5.history['accuracy'], label='Train Accuracy', color='hotpink')
plt.plot(history5.history['val_accuracy'], label='Val Accuracy', color='cadetblue')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Evolución de la Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

###############################################################################
"""
MODELO 6: LSTM con datos del 2017, 2018, 2019
"""
# mismos pasos para transofrmar los datos que en el Modelo 5 pero primero reducimos el dataset
# --------------------------
# 4. Convertir en secuencia
# --------------------------

datos_filtrados = datos_processed[datos_processed['ano'].isin([2017, 2018, 2019])].copy()

np.random.seed(13)
random.seed(13)
tf.random.set_seed(13)

X_seq6 = []
y_seq6 = []
años = [2017, 2018, 2019]
n_features = datos_features.shape[1]


for pid, group in datos_filtrados.groupby('id_asg'):
    group = group.groupby('ano').mean(numeric_only=True).reindex(años).fillna(0)

    # Asegurar columnas
    for col in feature_names:
        if col not in group.columns:
            group[col] = 0
    group = group[feature_names]

    features = group.values  # (3 años, n_features)

    label_row = datos_filtrados[(datos_filtrados['id_asg'] == pid) & (datos_filtrados['ano'] == group.index.max())]
    label = label_row['baja_prox_ano'].values[0] if not label_row.empty else 0

    X_seq6.append(features)
    y_seq6.append(label)

X_seq6 = np.stack(X_seq6)
y_seq6 = np.array(y_seq6)

print("Shape de X:", X_seq6.shape)  # (n_polizas, 3, n_features)
print("Shape de y:", y_seq6.shape)

# --------------------------
# 5. Train/test split
# -------------------------
np.random.seed(13)
random.seed(13)
tf.random.set_seed(13)

X_train6, X_test6, y_train6, y_test6 = train_test_split(X_seq6, y_seq6, test_size=0.2, random_state=42)

class_weights6 = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train6),
    y=y_train6
)
class_weights6 = dict(enumerate(class_weights6))
print(class_weights6)

# -----------------------------
# 6. Red Neuronal
# -----------------------------
timesteps = X_seq6.shape[1] # = 3 (años)
features = X_seq6.shape[2]  # = 67 (features por año)
d1,d2, d3 =(0.2,0.2, 0.2) # dropout

model6 = Sequential()
model6.add(LSTM(32, input_shape=(timesteps, features), return_sequences=True)) #
model6.add(Dropout(d1))
model6.add(LSTM(32, return_sequences=False))
model6.add(Dropout(d2))
model6.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
model6.add(Dropout(d3))
model6.add(Dense(1, activation='sigmoid'))

model6.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# -----------------------------
# 7. Entrenamiento
# -----------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

np.random.seed(13)
random.seed(13)
tf.random.set_seed(13)

history6 = model6.fit(
    X_train6, y_train6,
    validation_split=0.2,
    epochs=50,
    batch_size=64,
    class_weight=class_weights6,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# 8. Evaluación
# -----------------------------
loss6, acc6 = model6.evaluate(X_test6, y_test6)
print(f"Test Loss: {loss6:.4f}")
print(f"Test Accuracy: {acc6:.4f}")

# Predicción de probabilidades
y_proba6= model6.predict(X_test6)#.ravel()
y_pred6 = (y_proba6 >= 0.62).astype(int)


# Matriz de confusión
cm6=confusion_matrix(y_test6, y_pred6)
# Visualmente:
ax=sns.heatmap(cm6, annot=True, fmt="d",  cmap="BuPu", xticklabels=[0,1], yticklabels=[0,1])
ax.set_xlabel("Predicción")
ax.set_ylabel("Real")
ax.set_title("Modelo 6: Matriz de confusión")

# Métricas
print(classification_report(y_test6, y_pred6, digits=3))
print("AUC-ROC:", roc_auc_score(y_test6, y_proba6))

# -----------------------------
# 9. Gráficas
# -----------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history6.history['loss'], label='Train Loss', color='hotpink')
plt.plot(history6.history['val_loss'], label='Val Loss', color='cadetblue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Evolución de la pérdida')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history6.history['accuracy'], label='Train Accuracy', color='hotpink')
plt.plot(history6.history['val_accuracy'], label='Val Accuracy', color='cadetblue')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Evolución de la Accuracy')
plt.legend()

plt.tight_layout()
plt.show()