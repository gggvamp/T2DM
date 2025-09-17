import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import joblib

# === 1. Cargar datos ===
df = pd.read_excel("DB_DMT2_02_06_25_victor.xlsx", sheet_name="CSV_rev3")
df['Clase_codificada'] = df['Clase'].map({'S': 0, 'D': 1, 'C': 2})
df = df[df['Clase_codificada'].notnull()]
df = df[df['Complicaciones (0: sin, 1: micro, 2: macro)'].notnull()]

# === 2. Selección de variables importantes (sin fuga de datos) ===
top_features = [
    "Hipertenso",
    "Hb. Glic A1c",
    "Años con DMT2",
    "Colesterol Total",
    "IL_18",
    "GLUTATION_REDUCTASA",
    "HOMA-IR",  # reemplazo de 'Glucosa'
    "DM2 Familia (SI/NO)",
    "Edad",
    "Interpretacion HOMA-IR (1 sin r, 2-sr, 3. r)"
]


X = df[top_features]
y = df['Clase_codificada']

# === 3. Separar datos ===
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# === 4. Escalado ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 5. Entrenar modelo principal con GridSearch ===
param_grid = {
    'n_estimators': [100],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid.fit(X_train_scaled, y_train)
modelo = grid.best_estimator_

# === 6. Entrenar modelo secundario (complicaciones si clase = C) ===
df_c = df[df['Clase_codificada'] == 2]
X_c = df_c[top_features]
y_c = df_c['Complicaciones (0: sin, 1: micro, 2: macro)']
X_c_scaled = scaler.transform(X_c)
modelo_comp = RandomForestClassifier(random_state=42).fit(X_c_scaled, y_c)

# === 7. Predicción con riesgos para un paciente aleatorio ===
paciente_idx = df.sample(1).index[0]
paciente = df.loc[[paciente_idx]]
paciente_datos = scaler.transform(paciente[top_features])
proba_clase = modelo.predict_proba(paciente_datos)[0]
pred_clase = modelo.predict(paciente_datos)[0]
clase_nombre = ['S', 'D', 'C'][pred_clase]

print(f"\n Paciente ID: {paciente_idx}")
print(f"Diagnóstico más probable: {clase_nombre} ({pred_clase})")
print(f" Riesgo de estar sano (S): {proba_clase[0]*100:.1f}%")
print(f" Riesgo de ser diabético (D): {proba_clase[1]*100:.1f}%")
print(f" Riesgo de estar complicado (C): {proba_clase[2]*100:.1f}%")

if pred_clase == 2:
    proba_comp = modelo_comp.predict_proba(paciente_datos)[0]
    clases_presentes = modelo_comp.classes_
    etiquetas_dict = {0: 'Sin complicaciones', 1: 'Microvasculares', 2: 'Macrovasculares'}
    pred_comp = modelo_comp.predict(paciente_datos)[0]
    print(f"\nTipo más probable de complicación: {etiquetas_dict[pred_comp]}")
    for i, clase in enumerate(clases_presentes):
        nombre = etiquetas_dict.get(clase, f"Clase {clase}")
        print(f"Riesgo de {nombre}: {proba_comp[i]*100:.1f}%")
