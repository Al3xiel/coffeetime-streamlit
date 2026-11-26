# ☕ CoffeeTime Café – Predicción de retorno de clientes

Aplicación web en Streamlit que estima si un cliente de CoffeeTime Café regresará durante los próximos 30 días, mostrando la clase predicha y la probabilidad de retorno.

## Dataset

`clientes_historicos.csv` (1,500 clientes, datos sintéticos). Variable objetivo: `regresa_30_dias` (1 = regresa, 0 = no regresa).

Variables usadas: edad, visitas_ultimos_90_dias, gasto_promedio, dias_desde_ultima_visita, canal_compra, producto_favorito, usa_programa_fidelidad, descuentos_utilizados, calificacion_promedio, frecuencia_compras. `cliente_id` no se usa como variable.

## Modelo

Pipeline de scikit-learn (One-Hot Encoding para las variables categóricas + Random Forest Classifier) guardado en `model.joblib`.

## Cómo ejecutar

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # Windows (PowerShell)
pip install -r requirements.txt
python train_model.py           # entrena y guarda model.joblib
streamlit run app.py            # abre http://localhost:8501
```

## Estructura

```
coffeetime-streamlit/
├── clientes_historicos.csv
├── train_model.py
├── app.py
├── model.joblib
├── requirements.txt
└── README.md
```
