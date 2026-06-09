# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 21:14:43 2026

@author: Gerardo
"""

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_validate,
    RepeatedStratifiedKFold
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ==========================================================
# CARGAR BASE
# ==========================================================

df = pd.read_excel(
    "DiabetesMellitus2_Database_12-29-2025.xlsx",
    sheet_name="DM2_DB"
)

df.columns = df.columns.str.strip()

# ==========================================================
# STAGE 1
# Healthy / Diabetic / Complicated
# ==========================================================

df["Clase_codificada"] = df["Class"].map({

    "S": 0,
    "D": 1,
    "C": 2

})

df = df[df["Clase_codificada"].notnull()]

features_stage1 = [

    "Hypertension",
    "HbA1c",
    "Years with DM2",
    "Total Cholesterol",
    "IL_18",
    "Glutathione Reductase (GR)",
    "HOMA-IR",
    "DM2 Family History",
    "Age"

]

X1 = df[features_stage1]
y1 = df["Clase_codificada"]

X1_train, X1_test, y1_train, y1_test = train_test_split(

    X1,
    y1,

    test_size=0.20,

    stratify=y1,

    random_state=42

)

pipe_stage1 = Pipeline([

    ("scaler", StandardScaler()),

    ("rf", RandomForestClassifier(
        class_weight="balanced",
        random_state=42
    ))

])

param_grid_stage1 = {

    "rf__n_estimators":[100,500],

    "rf__max_depth":[
        5,
        10,
        None
    ],

    "rf__min_samples_split":[
        2,
        5
    ]
}

grid_stage1 = GridSearchCV(

    pipe_stage1,

    param_grid_stage1,

    cv=5,

    scoring="f1_macro",

    n_jobs=-1

)

grid_stage1.fit(
    X1_train,
    y1_train
)

modelo_stage1 = grid_stage1.best_estimator_

print("\n====================")
print("STAGE 1")
print("====================")

print(grid_stage1.best_params_)

# ==========================================================
# STAGE 2
# MICRO VS MACRO
# ==========================================================

df_c = df[
    df["Clase_codificada"] == 2
].copy()

df_c = df_c[
    df_c["Complications"].isin([1,2])
]

features_stage2 = [

    "Age",

    "Years with DM2",

    "HbA1c",

    "Creatinine",

    "Urea",

    "IL_6",

    "IL_10",

    "IL_18",

    "TNF_alfa",

    "miR-21",

    "miR-126",

    "AOPP",

    "LPO",

    "NO",

    "GSH",

    "GSSG_GSH",

    "Glutathione Reductase (GR)"

]

X2 = df_c[features_stage2]

y2 = df_c["Complications"]

pipe_stage2 = Pipeline([

    ("scaler", StandardScaler()),

    ("rf", RandomForestClassifier(
        class_weight="balanced",
        random_state=42
    ))

])

param_grid_stage2 = {

    "rf__n_estimators":[100,500],

    "rf__max_depth":[
        3,
        5,
        None
    ],

    "rf__min_samples_split":[
        2,
        5
    ]
}

grid_stage2 = GridSearchCV(

    pipe_stage2,

    param_grid_stage2,

    cv=5,

    scoring="f1_macro",

    n_jobs=-1

)

grid_stage2.fit(
    X2,
    y2
)

modelo_stage2 = grid_stage2.best_estimator_

print("\n====================")
print("STAGE 2")
print("====================")

print(grid_stage2.best_params_)

# ==========================================================
# PACIENTE ALEATORIO
# ==========================================================

paciente_idx = df.sample(
    1,
    random_state=None
).index[0]

paciente = df.loc[[paciente_idx]]

print("\n====================")
print("PACIENTE")
print("====================")

print("Código:", paciente["Codigo"].values[0])

# ==========================================================
# STAGE 1 PREDICCIÓN
# ==========================================================

proba_stage1 = modelo_stage1.predict_proba(
    paciente[features_stage1]
)[0]

pred_stage1 = modelo_stage1.predict(
    paciente[features_stage1]
)[0]

etiquetas_stage1 = {

    0:"Healthy",

    1:"Diabetic",

    2:"Complicated"

}

print("\nStage 1")

print(
    f"Healthy: {proba_stage1[0]*100:.2f}%"
)

print(
    f"Diabetic: {proba_stage1[1]*100:.2f}%"
)

print(
    f"Complicated: {proba_stage1[2]*100:.2f}%"
)

print(
    f"Predicción: {etiquetas_stage1[pred_stage1]}"
)

# ==========================================================
# STAGE 2
# ==========================================================

if pred_stage1 == 2:

    proba_stage2 = modelo_stage2.predict_proba(
        paciente[features_stage2]
    )[0]

    pred_stage2 = modelo_stage2.predict(
        paciente[features_stage2]
    )[0]

    etiquetas_stage2 = {

        1:"Microvascular",

        2:"Macrovascular"

    }

    print("\nStage 2")

    clases = modelo_stage2.classes_

    for i,c in enumerate(clases):

        print(
            f"{etiquetas_stage2[c]}: "
            f"{proba_stage2[i]*100:.2f}%"
        )

    print(
        f"Predicción final: "
        f"{etiquetas_stage2[pred_stage2]}"
    )
    
    ##################################################
    
   

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=100,
    random_state=42
)

scores = cross_validate(

    modelo_stage2,

    X2,
    y2,

    cv=cv,

    scoring={

        "accuracy":"accuracy",

        "precision":"precision_macro",

        "recall":"recall_macro",

        "f1":"f1_macro"

    },

    n_jobs=-1
)

print("\n====================")
print("STAGE 2 VALIDATION")
print("====================")

print(
    f"Accuracy : {scores['test_accuracy'].mean():.4f}"
)

print(
    f"Precision: {scores['test_precision'].mean():.4f}"
)

print(
    f"Recall   : {scores['test_recall'].mean():.4f}"
)

print(
    f"F1 Macro : {scores['test_f1'].mean():.4f}"
)

print(
    f"Accuracy STD : "
    f"{scores['test_accuracy'].std():.4f}"
)

print(
    f"F1 STD       : "
    f"{scores['test_f1'].std():.4f}"
)

modelo_stage2 = grid_stage2.best_estimator_

rf2 = modelo_stage2.named_steps["rf"]

importance_stage2 = pd.DataFrame({

    "Variable": features_stage2,

    "Importance": rf2.feature_importances_

})

importance_stage2 = importance_stage2.sort_values(
    "Importance",
    ascending=False
)

print("\n====================")
print("STAGE 2 FEATURE IMPORTANCE")
print("====================")

print(importance_stage2)


