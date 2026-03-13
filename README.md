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
| Modelo              | Parámetros | Train Acc | Val Acc | Test Acc | F1 macro(test) |
| ------------------- | ---------- | --------- | ------- | -------- | -------------- |
| Logistic Regression |   |   |  | 74% | 0.72 |
| XGBoost             |   |   |  | 84% | 0.81 |
| Random Forest       | 16592 | 100% | 97.67% | 96.8% | 0.93 |
| Modelo lineal (Regresión logistica) | 33 | 97.71% | 98.67% | 96.67% | 0.9325 |
| Modelo ML (Árbol de decisión) | 57 | 93.36% | 88.33% | 90.33% | 0.8577 |
| Modelo red neuronal | 33 | 96,79% | 98,00% | 96,67% | 0.9325 |
| Modelo complejo 1   | 931 | 99,43% | 98,00% | 97,67% | 0.9459 |
| Modelo complejo 2   | 2883 | 99,64% | 98,33% | 97,00% | 0.9323 |
| Modelo complejo 3   | 535 | 99,07% | 99,33% | 97,00% | 0.9393 |
| Modelo complejo 4   | 9859 | 99,57% | 98,00% | 97,67% | 0.9397 |
| Modelo complejo 5   | 3363 | 99,00% | 96,33% | 97,67% | 0.9436 |


### 3.1. Referencias
Según la documentación del dataset en Kaggle, modelos tradicionales como Logistic Regression se espera que alcancen una precisión aproximadamente de un 74% de accuracy, mientras que métodos de boosting como XGBoost en torno al 84%.
Algunos usuarios han reportado resultados superiores utilizando Random Forest, llegando hasta 96% de accuracy, lo que sugiere que el dataset presenta patrones altamente separables.

### 3.2. Métricas
El Accuracy mide el porcentaje total de aciertos sobre el total de casos, indicando qué tan bien predice el modelo en general; sin embargo, en problemas de salud laboral como el burnout, el F1-Macro es más crítico porque calcula la media del rendimiento de cada categoría (Bajo, Medio, Alto) por separado. 
Al usar el F1-Macro, nos aseguramos de que el modelo sea realmente capaz de detectar correctamente los casos de riesgo "Alto" y no se limite a ignorarlos para centrarse solo en la clase mayoritaria, ofreciendo así una visión mucho más equilibrada y realista del éxito del proyecto.

### 3.3. Conclusiones Modelos Simples
Tras experimentar con tres aproximaciones distintas (Lineal, Machine Learning y Deep Learning), hemos obtenido las siguientes conclusiones:

- La Regresión Logística destaca por su increíble eficiencia. Con solo 33 parámetros, iguala el rendimiento de modelos mucho más pesados, siendo la opción más sólida para una implementación rápida.

- El Árbol de Decisión, a pesar de tener un Accuracy menor, es nuestra elección recomendada para la detección de burnout alto. Su capacidad para no dejar escapar casos críticos (Recall 0.95) lo hace superior en un entorno de prevención de riesgos.

- La Red Neuronal ha demostrado una generalización perfecta. La paridad entre los resultados de entrenamiento y test confirma que los datos han sido preprocesados y escalados correctamente, permitiendo una convergencia limpia del modelo.

Resultado final: Hemos logrado superar los modelos de referencia iniciales, pasando de un 74% de acierto base a un 96.67%, utilizando arquitecturas simplificadas y optimizadas

### 3.4. Conclusiones Modelos Complejos

Tras explorar distintas arquitecturas de redes neuronales con mayor profundidad y número de parámetros, hemos analizado cómo influye la complejidad del modelo en la capacidad de aprendizaje y generalización del sistema.

El Modelo Complejo 1 representa el mejor equilibrio entre rendimiento y tamaño del modelo. Con una arquitectura de dos capas ocultas (32 y 16 neuronas), consigue el mejor resultado en F1-Macro (0.9459) y uno de los mayores accuracy en test (97.67%), demostrando que una red moderadamente profunda es suficiente para capturar las relaciones presentes en el dataset.

El Modelo Complejo 3, basado en una arquitectura más profunda pero con menos neuronas por capa, muestra un comportamiento muy estable durante el entrenamiento. Su alto accuracy en validación (99.33%) indica una excelente capacidad de generalización, confirmando que aumentar la profundidad puede mejorar la capacidad de representación del modelo sin incrementar excesivamente el número de parámetros.

El Modelo Complejo 5 introduce técnicas de regularización mediante Dropout, con el objetivo de reducir el riesgo de sobreajuste en arquitecturas más grandes. Los resultados obtenidos muestran que el modelo mantiene un rendimiento muy competitivo (97.67% de accuracy en test y F1-Macro 0.9436), demostrando que la regularización permite entrenar redes más profundas sin pérdida significativa de generalización.

Sin embargo, el análisis global de resultados muestra que aumentar el número de parámetros más allá de cierto punto no aporta mejoras sustanciales en el rendimiento. Incluso modelos muy grandes, como el Modelo Complejo 4, no superan de forma clara a arquitecturas más simples.

Resultado final: Los experimentos confirman que el problema de clasificación del burnout presenta patrones relativamente bien definidos. Por ello, arquitecturas de complejidad moderada como el Modelo Complejo 1 son capaces de capturar eficazmente la información del dataset, alcanzando resultados cercanos al 98% de accuracy, sin necesidad de utilizar redes excesivamente grandes o costosas computacionalmente.

### 3.5. Análisis de Complejidad del Modelo (Bias–Variance Tradeoff)

Con el objetivo de analizar cómo afecta la complejidad del modelo al rendimiento, se ha realizado un estudio del Bias–Variance Tradeoff utilizando las distintas arquitecturas evaluadas. En el gráfico se representan tres componentes clave: el error de entrenamiento (bias), la diferencia entre error de entrenamiento y test (varianza) y el error total en test.

La Red Simple presenta el mayor error de entrenamiento, lo que indica un bias relativamente alto debido a su baja complejidad. Sin embargo, la diferencia entre entrenamiento y test es prácticamente inexistente, lo que demuestra una gran estabilidad y ausencia de sobreajuste.

A medida que aumentamos la complejidad del modelo, como ocurre en Modelo C3 y Modelo C1, el error de entrenamiento disminuye significativamente. Esto indica que las redes son capaces de aprender patrones más complejos presentes en los datos. En estos casos también se observa una reducción del error total en test, lo que confirma una mejora en la capacidad de generalización.

El Modelo C2 muestra un ligero incremento en la varianza, reflejando una mayor diferencia entre los resultados de entrenamiento y test. Este comportamiento sugiere que, aunque el modelo aprende bien los datos de entrenamiento, comienza a aparecer una ligera tendencia al sobreajuste.

Por su parte, el Modelo C5 consigue mantener un buen equilibrio entre capacidad de aprendizaje y generalización gracias al uso de regularización mediante Dropout, lo que reduce la varianza y estabiliza el rendimiento en datos no vistos.

Finalmente, el Modelo C4, que es la arquitectura con mayor número de parámetros, no muestra mejoras claras respecto a modelos menos complejos. Esto confirma que incrementar excesivamente la complejidad del modelo no siempre produce mejores resultados, especialmente cuando el dataset ya presenta patrones relativamente bien definidos.

### 3.6. Modelo Final Seleccionado

A partir de los resultados obtenidos y del análisis de complejidad realizado, seleccionamos el Modelo Complejo 1 como modelo final del proyecto.

Esta arquitectura ofrece el mejor equilibrio entre rendimiento y complejidad, alcanzando uno de los mayores accuracy en test (97.67%) y el mejor F1-Macro (0.9459) entre las redes neuronales evaluadas, sin requerir un número excesivo de parámetros.

En comparación con arquitecturas más grandes, el Modelo Complejo 1 mantiene una alta capacidad de generalización sin introducir un aumento significativo en la varianza, lo que lo convierte en una solución eficiente y robusta para la clasificación del riesgo de burnout.

Conclusión final: Una arquitectura moderadamente profunda es suficiente para capturar los patrones presentes en el dataset, permitiendo alcanzar resultados cercanos al 98% de accuracy sin necesidad de utilizar redes excesivamente grandes o computacionalmente costosas.

## 4. Estructura del proyecto
- notebooks/: Contiene el Análisis Exploratorio de Datos (EDA).
- modelos/: Modelos del proyecto.
- data/: Datos del proyecto.
- requirements.txt: Librerías necesarias.
