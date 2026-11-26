"""Entrena el modelo de retorno de clientes de CoffeeTime Café y lo guarda en model.joblib."""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "regresa_30_dias"
NUMERIC = [
    "edad",
    "visitas_ultimos_90_dias",
    "gasto_promedio",
    "dias_desde_ultima_visita",
    "usa_programa_fidelidad",
    "descuentos_utilizados",
    "calificacion_promedio",
    "frecuencia_compras",
]
CATEGORICAL = ["canal_compra", "producto_favorito"]

# 1. Cargar datos históricos (cliente_id no se usa como variable)
df = pd.read_csv("clientes_historicos.csv")
X = df[NUMERIC + CATEGORICAL]
y = df[TARGET]

# 2. Separar entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Preprocesamiento + modelo en un solo pipeline
preprocess = ColumnTransformer(
    [
        ("num", "passthrough", NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ]
)
model = Pipeline(
    [
        ("preprocess", preprocess),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced", random_state=42
        )),
    ]
)
model.fit(X_train, y_train)

# 4. Evaluar
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, digits=3))
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# 5. Guardar el pipeline completo
joblib.dump(model, "model.joblib")
print("Modelo guardado en model.joblib")
