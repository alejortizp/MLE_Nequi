#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import mlflow
import pandas as pd
import logging
from dotenv import load_dotenv
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score, f1_score, fbeta_score, roc_curve, average_precision_score

# Importar funciones necesarias desde tu módulo de funciones
from NB_funciones import preprocess_data, log_info, log_error, CargarDatos


# In[10]:


# Configuración del modelo
CONFIG = {
    "data": {
        "dataset_name": "customer_support_twitter_twcs",
        "text_column": "text",
        "batch_size": 100  # Tamaño de lote para predicciones
    },
    "mlflow": {
        "experiment_name": "experimento_nuevo_final",
        "model_name": "modelo_nuevo"
    }
}


# In[ ]:


def cargar_modelo_mlflow():
    """
    Carga la última versión del modelo desde MLflow.

    Esta función construye la URI del modelo utilizando la configuración global y 
    lo recupera mediante `mlflow.sklearn.load_model()`. Si la carga falla, se registra 
    el error en los logs.

    Args:
        None (la función utiliza la configuración global `CONFIG` para obtener el nombre del modelo).

    Returns:
        sklearn model | None: Modelo cargado desde MLflow si la carga es exitosa, `None` en caso de error.

    Raises:
        Exception: Captura errores en la carga del modelo y los registra en los logs.
    """
    try:
        model_uri = f"models:/{CONFIG['mlflow']['model_name']}/latest"
        model = mlflow.sklearn.load_model(model_uri)
        log_info(f"Modelo cargado desde {model_uri}")
        return model
    except Exception as e:
        log_error(f"Error al cargar el modelo desde MLflow: {e}")
        return None


# In[ ]:


def hacer_predicciones_por_lotes(model, datos_nuevos, batch_size=100):
    """
    Realiza predicciones en lotes con el modelo cargado.

    Esta función divide los datos de entrada en bloques (`batch_size`) y genera predicciones 
    en cada iteración para optimizar el procesamiento en modelos grandes. Es útil cuando 
    el conjunto de datos es extenso y no puede procesarse de una sola vez.

    Args:
        model: Modelo entrenado que se utilizará para generar predicciones.
        datos_nuevos (pd.DataFrame): Conjunto de datos sobre el cual se desean obtener predicciones.
        batch_size (int, opcional): Tamaño del lote de datos procesados en cada iteración. Por defecto `100`.

    Returns:
        list | None: Lista con todas las predicciones generadas, `None` en caso de error.

    Raises:
        Exception: Captura errores en la inferencia y los registra en los logs.
    """
    try:
        predicciones_totales = []

        for i in range(0, len(datos_nuevos), batch_size):
            batch = datos_nuevos.iloc[i:i + batch_size]
            predicciones = model.predict(batch)
            predicciones_totales.extend(predicciones)

        return predicciones_totales
    except Exception as e:
        log_error(f"Error al hacer predicciones por lotes: {e}")
        return None


# In[ ]:


def evaluar_modelo(accuracy, umbral=0.80):
    """
    Evalúa si el modelo cumple con el umbral mínimo de precisión (`accuracy`).

    La función compara la precisión obtenida del modelo con un umbral predefinido y 
    devuelve `True` si la precisión cumple o supera dicho umbral, o `False` en caso contrario.

    Args:
        accuracy (float): Precisión obtenida del modelo a evaluar.
        umbral (float, opcional): Valor mínimo de precisión esperado para aprobar la evaluación. 
                                  Por defecto es `0.80` (80%).

    Returns:
        bool: `True` si la precisión del modelo es mayor o igual al umbral, `False` en caso contrario.
    """
    return accuracy >= umbral


# In[ ]:


def main():
    """
    Ejecuta el flujo de predicción en lotes con monitoreo.

    Este pipeline carga el modelo desde MLflow, obtiene nuevos datos, genera predicciones en lotes 
    y evalúa su rendimiento comparándolo con un umbral mínimo de precisión (`accuracy`).

    Returns:
        bool: `True` si el modelo cumple con el umbral de precisión, `False` en caso contrario.

    Raises:
        Exception: Captura errores en cualquier etapa del flujo y los registra en los logs.
    """
    try:
        log_info("Iniciando flujo de predicción en lotes...")

        # Cargar modelo
        model = cargar_modelo_mlflow()
        if model is None:
            log_error("No se pudo cargar el modelo. Deteniendo ejecución.")
            return False

        # Cargar datos nuevos
        ruta = CargarDatos(CONFIG["data"]["dataset_name"])
        datos_nuevos = ruta.cargar_csv()
        log_info(f"Datos nuevos cargados con {datos_nuevos.shape[0]} registros.")

        # Asegurar que el target sea int
        datos_nuevos['inbound'] = datos_nuevos['inbound'].astype('int')

        # Generar predicciones por lotes
        predicciones = hacer_predicciones_por_lotes(model, datos_nuevos['text'], CONFIG["data"]["batch_size"])
        predictions = [round(value) for value in predicciones]
        if predicciones is not None:
            log_info(f"Predicciones generadas exitosamente: {predicciones[:10]}")

        # Calcular accuracy
        accuracy = accuracy_score(datos_nuevos['inbound'], predictions)
        print(f"Accuracy: {accuracy:.2f}")

        # Evaluar si cumple con el umbral definido
        modelo_aprobado = evaluar_modelo(accuracy, umbral=0.80)  # Ajusta el umbral según necesidad
        print(f"¿Modelo cumple con el umbral? {modelo_aprobado}")

        return modelo_aprobado

    except Exception as e:
        log_error(f"Error crítico en el flujo de predicción por lotes: {e}")
        raise e


# In[15]:


if __name__ == "__main__":
    resultado = main()


# In[ ]:


#!jupyter nbconvert --to script NB_monitoreo_final.ipynb

