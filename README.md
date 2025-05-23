# MLE_Nequi
Prueba para proceso de Ingeniero Machine Learning Nequi

# Sistema MLOps para Clasificación de Texto

## Descripción

Este proyecto implementa un sistema completo de MLOps para clasificación de texto utilizando el dataset 'customer_support_twitter_twcs'. El sistema incluye pipelines automatizados para entrenamiento, predicción y monitoreo de modelos con integración completa de MLflow para el seguimiento de experimentos y gestión de modelos.

## Arquitectura del Sistema

El sistema está compuesto por tres componentes principales:

- **Entrenamiento** ('NB_entrenamiento_final.py'): Pipeline de entrenamiento y re-entrenamiento de modelos
- **Predicción** ('NB_prediccion_final.py'): Sistema de inferencia para nuevos datos
- **Monitoreo** ('NB_monitoreo_final.py'): Monitoreo del rendimiento del modelo en producción

## Tecnologías Utilizadas

- **Python 3.x**
- **MLflow**: Gestión de experimentos y modelos
- **Scikit-learn**: Pipeline de machine learning
- **CatBoost**: Algoritmo de clasificación
- **Pandas & NumPy**: Manipulación de datos
- **NLTK**: Procesamiento de lenguaje natural
- **TF-IDF**: Vectorización de texto

## Estructura del Proyecto

'''
├── NB_entrenamiento_final.py    # Pipeline de entrenamiento
├── NB_prediccion_final.py       # Sistema de predicción
├── NB_monitoreo_final.py        # Monitoreo del modelo
├── NB_funciones.py              # Módulo de funciones auxiliares
├── errores_entrenamiento.log    # Logs de errores
├── .env                         # Variables de entorno
└── README.md                    # Este archivo
'''

## Configuración e Instalación

### 1. Prerrequisitos

'''bash
pip install mlflow pandas numpy scikit-learn catboost nltk python-dotenv
'''

### 2. Configuración de Variables de Entorno

Crea un archivo '.env' en el directorio raíz:

'''env
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=experimento_nuevo_final
'''

### 3. Configuración de NLTK

'''python
import nltk
nltk.download('stopwords')
'''

### 4. Configuración de MLflow

Inicia el servidor de MLflow:

'''bash
mlflow server --host 0.0.0.0 --port 5000
'''

## Uso del Sistema

### Entrenamiento de Modelos

El script de entrenamiento ('NB_entrenamiento_final.py') ejecuta un pipeline completo que incluye:

- Carga y preprocesamiento de datos
- Configuración de experimentos en MLflow
- Entrenamiento del modelo con CatBoost y TF-IDF
- Evaluación y registro del modelo
- Comparación con modelos campeón (opcional)

'''bash
python NB_entrenamiento_final.py
'''

**Características principales:**
- Pipeline automatizado con Scikit-learn
- Registro automático de métricas y parámetros en MLflow
- Manejo robusto de errores con logging
- Configuración centralizada

### Predicción en Lotes

El script de predicción ('NB_prediccion_final.py') permite realizar inferencias sobre nuevos datos:

'''bash
python NB_prediccion_final.py
'''

**Características principales:**
- Carga automática del último modelo desde MLflow
- Procesamiento en lotes para optimizar rendimiento
- Manejo eficiente de memoria para datasets grandes

### Monitoreo del Modelo

El script de monitoreo ('NB_monitoreo_final.py') evalúa el rendimiento del modelo en producción:

'''bash
python NB_monitoreo_final.py
'''

**Características principales:**
- Evaluación automática del accuracy
- Comparación con umbrales de rendimiento
- Alertas cuando el modelo no cumple estándares mínimos

## Configuración del Sistema

### Parámetros del Modelo

'''python
CONFIG = {
    "data": {
        "dataset_name": "customer_support_twitter_twcs",
        "text_column": "text",
        "target_column": "inbound",
        "test_size": 0.3,
        "valid_size": 0.3,
        "random_state": 42
    },
    "mlflow": {
        "experiment_name": "experimento_nuevo_final"
    },
    "model": {
        "name": "modelo_nuevo",
        "version": "1.0.1",
        "champion_threshold": 0.5,
        "parameters": {
            "iterations": 50,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3,
            "random_seed": 42
        }
    }
}
'''

### Parámetros de TF-IDF

- **max_features**: 100
- **lowercase**: True
- **stop_words**: English stopwords
- **token_pattern**: 'r'\b\w+\b''

## Pipeline de Machine Learning

### 1. Preprocesamiento
- Limpieza de texto
- Tokenización
- Eliminación de stopwords
- Vectorización TF-IDF

### 2. Modelo
- **Algoritmo**: CatBoost Classifier
- **Características**: Manejo automático de características categóricas
- **Optimización**: Parámetros preconfigurados para clasificación de texto

### 3. Evaluación
- **Métricas principales**: Accuracy, Precision, Recall, F1-Score
- **Validación**: División train/test/validation
- **Monitoreo continuo**: Evaluación automática del rendimiento

## Logging y Monitoreo

### Sistema de Logs
- **Archivo**: 'errores_entrenamiento.log'
- **Nivel**: DEBUG
- **Formato**: Timestamp, nivel, mensaje

### Funciones de Logging
'''python
log_info("Mensaje informativo")
log_error("Mensaje de error")
'''

### Métricas Monitoreadas
- Accuracy del modelo
- Tiempo de entrenamiento
- Tamaño del dataset
- Parámetros del modelo

## Integración con MLflow

### Experimentos
- Seguimiento automático de experimentos
- Comparación de diferentes ejecuciones
- Versionado de modelos

### Registro de Modelos
- Almacenamiento automático de modelos entrenados
- Signature inference para input/output
- Ejemplos de entrada para validación

### Métricas y Parámetros
- Registro automático de hiperparámetros
- Tracking de métricas de evaluación
- Versionado de datasets

## Buenas Prácticas Implementadas

### MLOps
- **Reproducibilidad**: Seeds fijos y versionado de código
- **Monitoreo**: Evaluación continua del rendimiento
- **Automatización**: Pipelines completamente automatizados
- **Trazabilidad**: Logging completo de todas las operaciones

### Código
- **Modularidad**: Funciones separadas en módulos
- **Error Handling**: Manejo robusto de excepciones
- **Configuración**: Parámetros centralizados y fáciles de modificar
- **Documentación**: Docstrings detallados en todas las funciones

## Resolución de Problemas

### Errores Comunes

1. **Error de conexión con MLflow**
   - Verificar que el servidor MLflow esté ejecutándose
   - Comprobar la configuración de 'MLFLOW_TRACKING_URI'

2. **Error al cargar datos**
   - Verificar que el dataset esté disponible
   - Comprobar permisos de lectura del archivo

3. **Error en el modelo**
   - Verificar que el modelo esté registrado en MLflow
   - Comprobar la compatibilidad de versiones

### Logs de Debugging

Los logs se almacenan en 'errores_entrenamiento.log' con información detallada sobre:
- Errores de ejecución
- Estados del pipeline
- Métricas de rendimiento
- Información de debugging

## Contribución

Para contribuir al proyecto:

1. Fork el repositorio
2. Crea una branch para tu feature ('git checkout -b feature/nueva-funcionalidad')
3. Commit tus cambios ('git commit -am 'Añadir nueva funcionalidad'')
4. Push a la branch ('git push origin feature/nueva-funcionalidad')
5. Crea un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo 'LICENSE' para más detalles.

## Contacto

Para preguntas o soporte, por favor abre un issue en el repositorio del proyecto.

---

**Nota**: Este sistema está diseñado para entrenamiento y monitoreo automatizado de modelos de clasificación de texto. Asegúrate de tener configurado correctamente MLflow antes de ejecutar los scripts.