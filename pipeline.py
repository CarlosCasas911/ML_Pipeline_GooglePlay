# 📌 pipeline.py - Pipeline completo corregido y mejorado

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

# ===========================
# PASO 0: CONFIGURACIÓN
# ===========================
DATA_PATH = 'data/googleplaystore.csv'
OUTPUT_PATH = 'outputs'
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ===========================
# PASO 1: CARGAR DATOS
# ===========================
df = pd.read_csv(DATA_PATH)
print("📊 Dimensiones iniciales:", df.shape)

df.drop_duplicates(inplace=True)
df.replace("NaN", np.nan, inplace=True)
df.dropna(inplace=True)
print("✅ Después de eliminar duplicados y NaN:", df.shape)

# Conversión de tipos numéricos
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
df['Installs'] = df['Installs'].str.replace(r'[+,]', '', regex=True).astype(int)
df['Price'] = df['Price'].replace(r'[\$,]', '', regex=True).astype(float)

# Conversión de tamaño
def parse_size(size):
    if 'M' in size:
        return float(size.replace('M', ''))
    elif 'k' in size:
        return float(size.replace('k', '')) / 1024
    else:
        return np.nan

df['Size'] = df['Size'].apply(parse_size)
df.dropna(inplace=True)
print("✅ Después de procesar Size:", df.shape)

# ===========================
# PASO 2: TRANSFORMACIÓN
# ===========================
df['Last Updated'] = pd.to_datetime(df['Last Updated'], errors='coerce')
df['Last_Updated_Year'] = df['Last Updated'].dt.year

df['Android Ver'] = df['Android Ver'].str.extract(r'(\d+\.\d+)')  # ✔ Regex corregido
df['Android Ver'] = pd.to_numeric(df['Android Ver'], errors='coerce')

df = pd.get_dummies(df, columns=['Category', 'Type', 'Content Rating'], drop_first=False)
df.drop(columns=['App', 'Genres', 'Last Updated', 'Current Ver'], inplace=True)
df.dropna(inplace=True)
print("✅ Después de transformación y get_dummies:", df.shape)

# ===========================
# PASO 3: FEATURES EXTRA
# ===========================
df['is_free'] = (df['Price'] == 0).astype(int)
df['log_reviews'] = np.log1p(df['Reviews'])
df['rating_density'] = df['Rating'] / (df['Reviews'] + 1)
df['relative_size'] = df['Size'] / df['Size'].mean()

def categorizar_popularidad(x):
    if x <= 10000:
        return 0
    elif x <= 100000:
        return 1
    elif x <= 1000000:
        return 2
    else:
        return 3

df['popularity_class'] = df['Installs'].apply(categorizar_popularidad)
df.drop(columns=['Installs'], inplace=True)

print("✅ Después de ingeniería de características:", df.shape)
print("\n📌 Distribución de clases:\n", df['popularity_class'].value_counts())

# ===========================
# PASO 4: PARTICIÓN
# ===========================
if df['popularity_class'].nunique() < 2:
    print("❌ ERROR: No hay suficientes clases para entrenar. Verifica el dataset.")
    exit()

X = df.drop(columns=['popularity_class'])
y = df['popularity_class']

train_df, test_df = train_test_split(pd.concat([X, y], axis=1), test_size=0.2, random_state=42, stratify=y)

# Convertir booleanos a int
bool_cols_train = train_df.select_dtypes(include='bool').columns
bool_cols_test = test_df.select_dtypes(include='bool').columns
train_df[bool_cols_train] = train_df[bool_cols_train].astype(int)
test_df[bool_cols_test] = test_df[bool_cols_test].astype(int)

train_df.to_csv(f'{OUTPUT_PATH}/train.csv', index=False)
test_df.to_csv(f'{OUTPUT_PATH}/test.csv', index=False)

# ===========================
# PASO 5: MODELO
# ===========================
X_train = train_df.drop(columns=['popularity_class'])
y_train = train_df['popularity_class']
X_test = test_df.drop(columns=['popularity_class'])
y_test = test_df['popularity_class']

model = RandomForestClassifier(n_estimators=50, max_depth=None, max_features='log2', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

print("\n📊 MATRIZ DE CONFUSIÓN:\n", confusion_matrix(y_test, y_pred))
print("\n📊 REPORTE DE CLASIFICACIÓN:\n", classification_report(y_test, y_pred))

clases = sorted(model.classes_)
y_test_bin = label_binarize(y_test, classes=clases)

roc_auc_ovr = roc_auc_score(y_test_bin, y_proba, average=None)
roc_auc_macro = roc_auc_score(y_test_bin, y_proba, average='macro')
roc_auc_weighted = roc_auc_score(y_test_bin, y_proba, average='weighted')

print("\n✅ AUC por clase:", roc_auc_ovr)
print("✅ AUC macro:", roc_auc_macro)
print("✅ AUC ponderado:", roc_auc_weighted)

# ===========================
# PASO 6: CURVAS ROC
# ===========================
plt.figure(figsize=(8, 6))
for i, clase in enumerate(clases):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Clase {clase} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Curvas ROC por Clase')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig(f'{OUTPUT_PATH}/roc_curves.png')
plt.close()

# ===========================
# PASO 7: GUARDAR RESULTADOS
# ===========================
proba_df = pd.DataFrame(y_proba, columns=[f'prob_clase_{i}' for i in range(y_proba.shape[1])])
resultados = pd.DataFrame({'real': y_test.values, 'predicho': y_pred})
final_df = pd.concat([resultados, proba_df], axis=1)
final_df.to_csv(f'{OUTPUT_PATH}/predicciones_mejor_modelo.csv', index=False)

print("\n✅ Pipeline completado exitosamente. Archivos guardados en 'outputs/'.")
