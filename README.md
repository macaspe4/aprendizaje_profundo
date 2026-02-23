# Análisis de Agotamiento y Aislamiento Social (2026)
Este proyecto utiliza técnicas de Deep Learning para clasificar el nivel de burnout en trabajadores remotos.

## 1. Problema a resolver
El objetivo es predecir la categoría de agotamiento (Bajo, Medio, Alto) basándose en métricas de actividad y aislamiento. 

## 2. Dataset
- Nombre: Agotamiento de trabajo remoto y aislamiento social (2026)
- Variables de entrada (X): user_id, day_type, work_hours, screen_time_hours, meetings_count, breaks_taken, after_hours_work, app_switches, sleep_hours, task_completion, isolation_index, fatigue_score, burnout_score
- Variable de salida (y): Nivel de Burnout (burnout_risk)
Dimensiones: 2000 filas, 14 columnas

Ejemplo dataset:
| Variable | Valor | Descripción |
| :--- | :--- | :--- |
| **user_id** | 24 | Identificador único del empleado. |
| **day_type** | Weekday | Tipo de día (Laboral / Fin de semana). |
| **work_hours** | 8.48 | Horas totales trabajadas. |
| **screen_time_hours** | 7.23 | Horas de exposición a la pantalla. |
| **meetings_count** | 3 | Número de reuniones diarias. |
| **breaks_taken** | 5 | Descansos realizados durante la jornada. |
| **after_hours_work** | 0 | Trabajo realizado fuera del horario oficial. |
| **app_switches** | 63 | Cantidad de cambios entre aplicaciones. |
| **sleep_hours** | 7.31 | Horas de sueño registradas. |
| **task_completion** | 86.36 | Porcentaje de tareas finalizadas. |
| **isolation_index** | 4 | Índice percibido de aislamiento social. |
| **fatigue_score** | 5.15 | Puntuación de fatiga física/mental. |
| **burnout_score** | 24.01 | Puntuación numérica de agotamiento. |
| **burnout_risk (Target)** | **Low** | Categoría de riesgo (Clase a predecir). |


## 3. Estado del Arte
| Modelo              | Accuracy | F1-Macro | Tipo            |
| ------------------- | -------- | -------- | --------------- |
| Logistic Regression | 74%      | 0.72     | Baseline lineal |
| XGBoost             | 84%      | 0.81     | Boosting        |
| Random Forest       | 96.8%    | 0.93     | Ensemble        |
| MLP (objetivo)      | >95%     | >0.93    | Deep Learning   |

| Modelo              | Parámetros | Train Acc | Val Acc | Test Acc | F1 macro(test) |
| ------------------- | ---------- | --------- | ------- | -------- | -------------- |
| Logistic Regression |   |   |   | 74%  | 0.72 |
| XGBoost             |   |   |   |  84% | 0.81 |
| Random Forest       | 16592 | 100% | 97.67% | 96.8% | 0.93 |
| Modelo lineal (Regresión logistica) | 33 | 97.71% | 98.67% | 96.67% | 0.9325 |
| Modelo ML (Árbol de decisión) | 57 | 93.36% | 88.33% | 90.33% | 0.8577 |
| Modelo red neuronal |  |    |   |   |   |


### 3.1. Referencias
Según la documentación del dataset en Kaggle, modelos tradicionales como Logistic Regression se espera que alcancen una precisión aproximadamente de un 74% de accuracy, mientras que métodos de boosting como XGBoost en torno al 84%.
Algunos usuarios han reportado resultados superiores utilizando Random Forest, llegando hasta 96% de accuracy, lo que sugiere que el dataset presenta patrones altamente separables.

### 3.2. Métricas
El Accuracy mide el porcentaje total de aciertos sobre el total de casos, indicando qué tan bien predice el modelo en general; sin embargo, en problemas de salud laboral como el burnout, el F1-Macro es más crítico porque calcula la media del rendimiento de cada categoría (Bajo, Medio, Alto) por separado. 
Al usar el F1-Macro, nos aseguramos de que el modelo sea realmente capaz de detectar correctamente los casos de riesgo "Alto" y no se limite a ignorarlos para centrarse solo en la clase mayoritaria, ofreciendo así una visión mucho más equilibrada y realista del éxito del proyecto.


## 4. Estructura del proyecto
- notebooks/: Contiene el Análisis Exploratorio de Datos (EDA).
- data/: Datos del proyecto.
- requirements.txt: Librerías necesarias.
