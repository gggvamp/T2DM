# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 13:25:51 2026

@author: Gerardo
"""

# -*- coding: utf-8 -*-
"""
Modelo jerárquico DM2 — Clasificación S / D / C + Complicaciones
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt


# =========================
# 1) CARGAR DATOS
# =========================
df = pd.read_excel(
    "DiabetesMellitus2_Database_12-29-2025.xlsx",
    sheet_name="DM2_DB"
)

print("\nFilas iniciales:", df.shape)

# =========================
# 2) CODIFICAR CLASE
# =========================
df["Class"] = df["Class"].astype(str).str.strip().str.upper()

df["Clase_codificada"] = df["Class"].map({"S": 0, "D": 1, "C": 2})

print("\nConteo Clase_codificada:")
print(df["Clase_codificada"].value_counts(dropna=False))

# quitar filas sin clase
df = df[df["Clase_codificada"].notna()]

# asegurar columna correcta de complicaciones
print("\nColumnas disponibles:")
print(df.columns)

df = df[df["Complications"].notna()]

print("\nFilas después de filtrar complicaciones:", df.shape)


# =========================
# 3) VARIABLES USADAS
# =========================
top_features = [
    "Hypertension",
    "HbA1c",
    "Years with DM2",
    "Total Cholesterol",
    "IL_18",
    "Glutathione Reductase (GR)",
    "HOMA-IR",
    "DM2 Family History",
    "Age",
    "Creatinine",
    "IL_6",
    "LPO"
]

X = df[top_features]
y = df["Clase_codificada"]


# =========================
# 4) TRAIN / TEST
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    stratify=y,
    test_size=0.2,
    random_state=42
)

# =========================
# 5) ESCALADO
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================
# 6) RANDOM FOREST + GRIDSEARCH
# =========================
param_grid = {
    "n_estimators": [100],
    "max_depth": [5, 8, None],
    "min_samples_split": [2, 5]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5
)

grid.fit(X_train_scaled, y_train)
modelo = grid.best_estimator_

####################################################################
# === FEATURE IMPORTANCE ===
import numpy as np

importancias = modelo.feature_importances_
variables = X.columns

feature_importance = sorted(
    zip(variables, importancias),
    key=lambda x: x[1],
    reverse=True
)

print("\n=== FEATURE IMPORTANCE (Random Forest) ===")
for v, imp in feature_importance:
    print(f"{v:25s}  ->  {imp:.4f}")
    
import matplotlib.pyplot as plt

vars_sorted = [v for v, _ in feature_importance]
imps_sorted = [imp for _, imp in feature_importance]

plt.figure(figsize=(6.5, 4.5))
plt.barh(vars_sorted, imps_sorted)
plt.gca().invert_yaxis()

plt.xlabel("Feature importance", fontsize=10)
plt.title("Feature Importance — Random Forest", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("feature_importance_RF.png", dpi=400, bbox_inches="tight")
plt.show()

##################################################################
# =========================
# 7) VALIDACIÓN CRUZADA
# =========================
scoring = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro"
}

cv_results = cross_validate(
    modelo,
    X, y,
    cv=5,
    scoring=scoring,
    n_jobs=-1
)

print("\n=== VALIDACIÓN CRUZADA (5-FOLD) ===")
print(f"Accuracy medio: {cv_results['test_accuracy'].mean():.3f}")
print(f"Precision macro: {cv_results['test_precision_macro'].mean():.3f}")
print(f"Recall macro:    {cv_results['test_recall_macro'].mean():.3f}")
print(f"F1 macro:        {cv_results['test_f1_macro'].mean():.3f}")


# =========================
# 8) REPORTE TEST
# =========================
y_pred = modelo.predict(X_test_scaled)

print("\nREPORTE CLASIFICACIÓN:")
print(classification_report(y_test, y_pred, target_names=["S", "D", "C"]))

print("\nMATRIZ DE CONFUSIÓN:")
cm = confusion_matrix(y_test, y_pred)
print(cm)


# =========================
# 9) AJUSTE UMBRAL C
# =========================
probas = modelo.predict_proba(X_test_scaled)

THRESHOLD_C = 0.45
y_pred_adj = []

for p in probas:
    if p[2] >= THRESHOLD_C:
        y_pred_adj.append(2)
    else:
        y_pred_adj.append(np.argmax(p))

print("\n=== MATRIZ AJUSTADA ===")
print(confusion_matrix(y_test, y_pred_adj))


# =========================
# 10) MODELO COMPLICACIONES
# =========================
df_c = df[df["Clase_codificada"] == 2]

X_c = df_c[top_features]
y_c = df_c["Complications"]

X_c_scaled = scaler.transform(X_c)

modelo_comp = RandomForestClassifier(random_state=42)
modelo_comp.fit(X_c_scaled, y_c)


    
    ################################################
# === Predicción PARA PACIENTE ALEATORIO (formato clínico) ===

paciente_idx = df.sample(1).index[0]
paciente = df.loc[[paciente_idx]]

paciente_datos = scaler.transform(paciente[top_features])

probas = modelo.predict_proba(paciente_datos)[0]
pred_clase = modelo.predict(paciente_datos)[0]

# probabilidades por clase
p_s = probas[0]
p_d = probas[1]
p_c = probas[2]

pred = ["S (Sano)", "D (Diabético)", "C (Complicado)"][pred_clase]

print(f"\nPaciente ID: {paciente_idx}")
print(f"Diagnóstico más probable: {pred}")

print(f"Riesgo de estar sano (S): {p_s*100:.1f}%")
print(f"Riesgo de ser diabético (D): {p_d*100:.1f}%")
print(f"Riesgo de estar complicado (C): {p_c*100:.1f}%")
    
# --- si cae en C, mostramos detalle ---
if pred_clase == 2:

    paciente_c = paciente[top_features]
    paciente_c_scaled = scaler.transform(paciente_c)

    proba_comp = modelo_comp.predict_proba(paciente_c_scaled)[0]
    clases_presentes = modelo_comp.classes_

    probas_dict = {}

    etiquetas = {
        0: "C-Sin",
        1: "C-Micro",
        2: "C-Macro"
    }

    for i, c in enumerate(clases_presentes):
        probas_dict[etiquetas[c]] = proba_comp[i]

    print("\nDesglose de complicaciones:")
    print(f"  Sin complicaciones: {probas_dict.get('C-Sin', 0)*100:.1f}%")
    print(f"  Microvasculares:   {probas_dict.get('C-Micro', 0)*100:.1f}%")
    print(f"  Macrovasculares:   {probas_dict.get('C-Macro', 0)*100:.1f}%")

clase_nombre = ["S", "D", "C"][pred_clase]

print(f"\nPaciente ID: {paciente_idx}")
print(f"Diagnóstico: {clase_nombre}")
print(f"Probabilidades: {probas}")

if pred_clase == 2:
    proba_comp = modelo_comp.predict_proba(paciente_datos)[0]
    print("\nRiesgos de complicación:", proba_comp)
    
# =========================
# 12) MATRIZ DE CONFUSIÓN (gráfica)
# =========================
labels_true = ["Healthy", "Diabetic", "Complicated"]
labels_pred = ["Healthy", "Diabetic", "Complicated"]

plt.figure(dpi=400)
plt.imshow(cm, cmap="Blues")
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=10)

plt.xticks(range(3), labels_pred, fontsize=11)
plt.yticks(range(3), labels_true, fontsize=11)

plt.title("Confusion Matrix", fontsize=11,fontweight="bold")
plt.xlabel("Predicted label", fontsize=10)
plt.ylabel("True label", fontsize=10)

for i in range(len(cm)):
    for j in range(len(cm[0])):
        plt.text(j, i, cm[i][j], ha="center", va="center", fontsize=12)

plt.tight_layout()
plt.savefig("confusion_matrix.tif", dpi=400, bbox_inches="tight")
plt.show()
##################################################################
probas = modelo.predict_proba(X_test_scaled)

# =========================
# ROC — MULTICLASS S / D / C
# =========================
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Definición de clases (consistentemente con el modelo)
classes = [0, 1, 2]      # 0=S, 1=D, 2=C
names   = {0:"Healthy", 1:"Diabetic", 2:"Complicated"}

# Convertir y_test a matriz binaria (uno vs resto)
y_test_bin = label_binarize(y_test, classes=classes)

print("\nShapes:")
print(" y_test_bin:", y_test_bin.shape)
print(" probas:", probas.shape)

fpr = {}
tpr = {}
roc_auc = {}

# Curvas por clase
for i, c in enumerate(classes):

    # Evitar el error: si la clase NO aparece en el test, se omite
    if y_test_bin[:, i].sum() == 0:
        print(f"⚠ Clase {names[c]} NO aparece en el test — se omite ROC.")
        continue

    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], probas[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Micro-average (solo si hay clases válidas)
if len(roc_auc) > 0:
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_test_bin.ravel(), probas.ravel()
    )
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# =========================
# GRAFICAR
# =========================
plt.figure(figsize=(7, 5), dpi=400)

color_map = {0:"blue", 1:"orange", 2:"red"}

for k, auc_val in roc_auc.items():

    if k == "micro":
        plt.plot(
            fpr[k], tpr[k],
            "k--",
            label=f"Micro-average (AUC = {auc_val:.2f})"
        )
    else:
        plt.plot(
            fpr[k], tpr[k],
            color=color_map[k],
            label=f"{names[k]} (AUC = {auc_val:.2f})"
        )

plt.plot([0, 1], [0, 1], "k:")

plt.xlabel("False Positive Rate", fontsize=11)
plt.ylabel("True Positive Rate", fontsize=11)
plt.title("ROC Curves — Multiclass (S / D / C)", fontsize=12, fontweight="bold")
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("ROC_multiclass_SDC.tif", dpi=400, bbox_inches="tight")
plt.show()

# =========================
# 12-B) MATRIZ DE CONFUSIÓN NORMALIZADA — MISMO FORMATO
# =========================

cm_norm = cm / cm.sum(axis=1, keepdims=True)

labels_true = ["Healthy", "Diabetic", "Complicated"]
labels_pred = ["Healthy", "Diabetic", "Complicated"]

plt.figure(dpi=400)
plt.imshow(cm_norm, cmap="Blues")

cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=10)

plt.xticks(range(3), labels_pred, fontsize=11)
plt.yticks(range(3), labels_true, fontsize=11)

plt.title("Confusion Matrix (Normalized)", fontsize=11,fontweight="bold")
plt.xlabel("Predicted label", fontsize=10)
plt.ylabel("True label", fontsize=10)

# valores dentro de cada celda (con 2 decimales)
for i in range(len(cm_norm)):
    for j in range(len(cm_norm[0])):
        plt.text(
            j, i,
            f"{cm_norm[i][j]:.2f}",
            ha="center",
            va="center",
            fontsize=12
        )

plt.tight_layout()
plt.savefig("confusion_matrix_normalized.tif", dpi=400, bbox_inches="tight")
plt.show()


###################Dibujar un árbol del modelo S / D / C###########
from sklearn import tree

plt.figure(figsize=(10,6))

tree.plot_tree(
    modelo.estimators_[0],
    feature_names=top_features,
    class_names=["Healthy","Diabetic","Complicated"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Representative tree of the Random Forest — Classification S/D/C", fontsize=11,fontweight="bold")

plt.tight_layout()
plt.savefig("TreeClassification SDC.tif", dpi=400, bbox_inches="tight")
plt.show()

########################Árbol SOLO para complicaciones (C)########


plt.figure(figsize=(10,6))

tree.plot_tree(
    modelo_comp.estimators_[0],
    feature_names=top_features,
    class_names=["Micro","Macro"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Representative tree — Complications in patients C", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("TreeComplications.tif", dpi=400, bbox_inches="tight")
plt.show()


# SHAP — interpretability module
# ==========================================
import shap
import matplotlib.pyplot as plt

print("\nGenerating SHAP explanations...")

# SHAP universal explainer
explainer = shap.Explainer(modelo, X_train_scaled)

# SHAP on test set
shap_values = explainer(X_test_scaled)

print("\nShapes:")
print(" shap_values:", shap_values.values.shape)
print(" X_test_scaled:", X_test_scaled.shape)

# =====================================================
# 1️⃣ SHAP summary plot (impact per feature) — CLASS C
# =====================================================
shap.summary_plot(
    shap_values.values[:, :, 2],      # class C
    X_test_scaled,
    feature_names=list(X.columns),
    show=False
)

# plt.title(
#     "Feature impact on predicting complication class (C)",
#     fontsize=12,
#     fontweight="bold"
#)

plt.tight_layout()
plt.savefig("SHAP_ClassC_summary.tif", dpi=400, bbox_inches="tight")
plt.show()

# =====================================================
# 2️⃣ SHAP bar plot (global importance) — CLASS C
# =====================================================
# --- Clase C = índice 2 ---
shap_c = shap_values.values[:, :, 2]

# mean(|shap|) por variable
mean_abs = np.mean(np.abs(shap_c), axis=0)

# desviación estándar (opcional, útil para paper)
std_abs = np.std(np.abs(shap_c), axis=0)

tabla_shap = pd.DataFrame({
    "Feature": X.columns,
    "Mean(|SHAP|)": mean_abs,
    "Std(|SHAP|)": std_abs
}).sort_values("Mean(|SHAP|)", ascending=False)

print("\n=== SHAP metrics for Class C (printed to console) ===\n")
print(tabla_shap.to_string(index=False))

shap.summary_plot(
    shap_values.values[:, :, 2],
    X_test_scaled,
    feature_names=list(X.columns),
    plot_type="bar",
    show=False
)

# plt.title(
#     "Global feature importance for complication prediction (Class C)",
#     fontsize=11,
#     fontweight="bold"
#)

plt.tight_layout()
plt.savefig("SHAP_ClassC_bar.tif", dpi=400, bbox_inches="tight")
plt.show()

####################################################################SDC
import shap
import numpy as np
import matplotlib.pyplot as plt

explainer = shap.Explainer(modelo, X_train_scaled)
shap_values = explainer(X_test_scaled)

feature_names = list(X.columns)

print("\nFormas de los objetos:")
print(" shap_values:", shap_values.values.shape)
print(" X_test_scaled:", X_test_scaled.shape)

# =========================
# FUNCIÓN AUXILIAR → imprime métricas
# =========================
def print_shap_metrics(shap_matrix, class_name):
    """
    shap_matrix: matriz [n_samples, n_features] para UNA clase
    """
    mean_abs = np.abs(shap_matrix).mean(axis=0)

    print(f"\n==============================")
    print(f"IMPORTANCIA SHAP — CLASE {class_name}")
    print(f"(promedio del valor absoluto)")
    print(f"==============================")

    for fname, val in sorted(zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True):
        print(f"{fname:30s}  ->  {val:.4f}")


# =========================
# CLASS 0 → S (Healthy)
# =========================
# =========================
# CLASS 0 → S (Healthy)
# =========================

# ---- SUMMARY PLOT ----
plt.figure(figsize=(8,5))

shap.summary_plot(
    shap_values.values[:, :, 0],
    X_test_scaled,
    feature_names=feature_names,
    show=False
)

ax = plt.gca()

# plt.title(
#     "Feature impact on predicting class S (Healthy)",
#     fontsize=12,
#     fontweight="bold"
# )

# etiquetas eje Y
for label in ax.get_yticklabels():
    label.set_fontsize(11)
    label.set_fontweight("bold")

# etiquetas eje X
for label in ax.get_xticklabels():
    label.set_fontsize(10)

plt.tight_layout()
plt.savefig("shap_summary_S.tiff", dpi=400, bbox_inches="tight")
plt.show()


# ---- BAR PLOT ----
plt.figure(figsize=(8,5))

shap.summary_plot(
    shap_values.values[:, :, 0],
    X_test_scaled,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)

ax = plt.gca()

# plt.title(
#     "Global feature importance — Class S (Healthy)",
#     fontsize=12,
#     fontweight="bold"
# )

# etiquetas eje Y
for label in ax.get_yticklabels():
    label.set_fontsize(11)
    label.set_fontweight("bold")

# etiquetas eje X
for label in ax.get_xticklabels():
    label.set_fontsize(10)

plt.tight_layout()
plt.savefig("shap_bar_S.tiff", dpi=400, bbox_inches="tight")
plt.show()


print_shap_metrics(shap_values.values[:, :, 0], "S (Healthy)")


# =========================
# CLASS 1 → D (Diabetes)
# =========================

# ---- SUMMARY PLOT ----
plt.figure(figsize=(8,5))

shap.summary_plot(
    shap_values.values[:, :, 1],
    X_test_scaled,
    feature_names=feature_names,
    show=False
)

ax = plt.gca()

# plt.title(
#     "Feature impact on predicting class D (Diabetes)",
#     fontsize=12,
#     fontweight="bold"
# )

# etiquetas eje Y
for label in ax.get_yticklabels():
    label.set_fontsize(11)
    label.set_fontweight("bold")

# etiquetas eje X
for label in ax.get_xticklabels():
    label.set_fontsize(10)

plt.tight_layout()
plt.savefig("shap_summary_D.tiff", dpi=400, bbox_inches="tight")
plt.show()


# ---- BAR PLOT ----
plt.figure(figsize=(8,5))

shap.summary_plot(
    shap_values.values[:, :, 1],
    X_test_scaled,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)

ax = plt.gca()

# plt.title(
#     "Global feature importance — Class D (Diabetes)",
#     fontsize=12,
#     fontweight="bold"
# )

# etiquetas eje Y
for label in ax.get_yticklabels():
    label.set_fontsize(11)
    label.set_fontweight("bold")

# etiquetas eje X
for label in ax.get_xticklabels():
    label.set_fontsize(10)

plt.tight_layout()
plt.savefig("shap_bar_D.tiff", dpi=400, bbox_inches="tight")
plt.show()


print_shap_metrics(shap_values.values[:, :, 1], "D (Diabetes)")


# =========================
# CLASS 2 → C (Complications)
# =========================
plt.figure(figsize=(8,5))

shap.summary_plot(
    shap_values.values[:, :, 2],
    X_test_scaled,
    feature_names=feature_names,
    show=False
)

ax = plt.gca()

# ---- título ----
# plt.title(
#     "Feature impact on predicting class C (Complications)",
#     fontsize=12,
#     fontweight="bold"
# )

# ---- etiquetas del eje Y (features) ----
for label in ax.get_yticklabels():
    label.set_fontsize(11)
    label.set_fontweight("bold")

# ---- eje X ----
for label in ax.get_xticklabels():
    label.set_fontsize(10)

plt.tight_layout()
plt.savefig("shap_summary_C.tiff", dpi=400, bbox_inches="tight")
plt.show()


# --- BAR PLOT ---
plt.figure(figsize=(8,5))

shap.summary_plot(
    shap_values.values[:, :, 2],
    X_test_scaled,
    feature_names=list(X.columns),
    plot_type="bar",
    show=False
)

# ---- format axis labels ----
plt.xlabel("Mean |SHAP value|\n(Average impact on model output)", fontsize=11)

ax = plt.gca()

# Y labels (variables)
for label in ax.get_yticklabels():
    label.set_fontsize(11)
    label.set_fontweight("bold")

# X labels
for label in ax.get_xticklabels():
    label.set_fontsize(10)

plt.tight_layout()
plt.savefig("SHAP_ClassC_bar.tif", dpi=400, bbox_inches="tight")
plt.show()



print_shap_metrics(shap_values.values[:, :, 2], "C (Complications)")

#####################################################################


