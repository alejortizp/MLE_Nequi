#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
import mlflow
import pandas as pd
import logging
from dotenv import load_dotenv

# Importar funciones necesarias desde tu módulo de funciones
from NB_funciones import preprocess_data, log_info, log_error, CargarDatos


# In[9]:


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
    Carga el modelo registrado en MLflow.

    Esta función recupera la última versión del modelo especificado en MLflow y lo carga
    utilizando `mlflow.sklearn.load_model`. Si ocurre un error, lo registra en los logs.

    Args:
        None (usa la configuración global `CONFIG` para obtener el nombre del modelo).

    Returns:
        sklearn model | None: Modelo cargado desde MLflow o `None` en caso de error.

    Raises:
        Exception: Captura errores durante la carga y los registra en los logs.
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


def hacer_predicciones(model, datos_nuevos):
    """
    Realiza predicciones con el modelo cargado.

    Esta función toma un modelo previamente entrenado y realiza predicciones sobre un conjunto 
    de datos nuevos. Si ocurre un error durante la ejecución, se captura y registra en los logs.

    Args:
        model: Modelo entrenado que se utilizará para generar predicciones.
        datos_nuevos: Datos de entrada sobre los cuales se desea obtener predicciones.

    Returns:
        np.ndarray | list | None: Array o lista con las predicciones generadas, 
                                  `None` si ocurre un error.

    Raises:
        Exception: Captura errores en la inferencia y los registra en los logs.

    """
    try:
        predicciones = model.predict(datos_nuevos)
        return predicciones
    except Exception as e:
        log_error(f"Error al hacer predicciones: {e}")
        return None

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


def main():
    """
    Ejecuta el flujo de predicción en lotes.

    Este pipeline carga el modelo desde MLflow, obtiene nuevos datos desde un archivo CSV 
    y genera predicciones por lotes para optimizar la inferencia en conjuntos de datos grandes.

    Returns:
        None: No devuelve valores explícitos, pero registra información relevante en los logs.

    Raises:
        Exception: Captura errores en cualquier etapa y los registra en los logs.
    """
    try:
        log_info("Iniciando flujo de predicción en lotes...")

        # Cargar modelo
        model = cargar_modelo_mlflow()
        if model is None:
            log_error("No se pudo cargar el modelo. Deteniendo ejecución.")
            return

        # Cargar datos nuevos
        ruta = CargarDatos(CONFIG["data"]["dataset_name"])
        datos_nuevos = ruta.cargar_csv()
        log_info(f"Datos nuevos cargados con {datos_nuevos.shape[0]} registros.")

        # Generar predicciones por lotes
        predicciones = hacer_predicciones_por_lotes(model, datos_nuevos['text'], CONFIG["data"]["batch_size"])
        if predicciones is not None:
            log_info(f"Predicciones generadas exitosamente: {predicciones[:10]}")  # Mostramos solo 10 ejemplos

    except Exception as e:
        log_error(f"Error crítico en el flujo de predicción por lotes: {e}")
        raise e


# In[13]:


if __name__ == "__main__":
    main()


# In[ ]:


#!jupyter nbconvert --to script NB_prediccion_final.ipynb

