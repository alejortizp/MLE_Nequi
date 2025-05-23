#!/usr/bin/env python
# coding: utf-8

# ## Script de entrenamiento y re-entrenamiento de modelos con prácticas MLOps.

# In[1]:


# Cargar variables de entorno
import os
import pandas as pd
import numpy as np
import mlflow
import logging
from dotenv import load_dotenv
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from catboost import CatBoostClassifier
import time
from mlflow.models.signature import infer_signature
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score, f1_score, fbeta_score, roc_curve, average_precision_score

# Importar funciones desde nuestro módulo de funciones
from NB_funciones import (
    CargarDatos,
    log_info, 
    log_error,
    setup_environment,
    preprocess_data,
    create_mlflow_experiment,
    evaluate_model,
    run_model_training_pipeline,
    data_drift_detection,
    model_performance_monitoring
)


# In[2]:


# Configuración del logging
logging.basicConfig(
    filename="errores_entrenamiento.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# In[3]:


# Configuración sin tracking_uri
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


# In[ ]:


def setup_mlflow():
    """
    Configura MLflow obteniendo o creando un experimento según su nombre.

    La función intenta recuperar un experimento existente en MLflow por su nombre.
    Si el experimento no existe, se crea uno nuevo. En caso de error en la conexión con MLflow, 
    se implementa una espera (`sleep`) antes de retornar `None`.

    Args:
        None (usa la configuración global `CONFIG` para obtener el nombre del experimento).

    Returns:
        str | None: Identificador del experimento (`experiment_id`) si la configuración es exitosa, 
                    `None` en caso de fallo.

    Raises:
        mlflow.exceptions.MlflowException: Maneja errores de conexión con MLflow y los registra en los logs.
    """
    experiment_name = CONFIG["mlflow"]["experiment_name"]

    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)

        if experiment is None:
            log_info(f"Creando nuevo experimento: {experiment_name}")
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id

        return experiment_id

    except mlflow.exceptions.MlflowException as e:
        log_error(f"Error al conectar con MLflow: {e}")
        time.sleep(5)
        return None



# In[ ]:


def main():
    """
    Ejecuta el pipeline completo de entrenamiento, minimizando conexiones innecesarias con MLflow.

    Este pipeline configura el entorno, carga y preprocesa datos, define parámetros del modelo,
    construye un pipeline de entrenamiento y maneja la comparación con un modelo campeón si existe.
    También registra el modelo en MLflow con sus métricas y artefactos relevantes.

    Returns:
        None: La función ejecuta el flujo completo sin retorno explícito.

    Raises:
        Exception: Captura errores en distintas etapas y los registra en los logs.
    """
    try:
        log_info("Iniciando pipeline de entrenamiento...")
        setup_environment()

        # Configurar MLflow con una sola conexión inicial
        experiment_id = setup_mlflow()
        if experiment_id is None:
            log_error("No se pudo establecer conexión con MLflow. Deteniendo ejecución.")
            return

        try:
            # Cargar datos
            ruta = CargarDatos(CONFIG["data"]["dataset_name"])
            data = ruta.cargar_csv()
            log_info(f"Dataset cargado con {data.shape[0]} registros.")
            logging.info(f"Dataset cargado con {data.shape[0]} registros.")
        except Exception as e:
            log_error(f"Error al cargar el dataset: {e}")
            return

        try:
            # Preprocesar datos
            text_column = CONFIG["data"]["text_column"]
            target_column = CONFIG["data"]["target_column"]
            data_splits = preprocess_data(data, text_column, target_column)
            logging.info(f"Datos preprocesados: {data_splits['X_train'].shape[0]} train, {data_splits['X_test'].shape[0]} test.")
        except Exception as e:
            log_error(f"Error en la preprocesación de datos: {e}")
            return

        try:
            # Definir parámetros
            english_stopwords = stopwords.words('english')
            catboost_params = CONFIG["model"]["parameters"]
            logging.info(f"Parámetros de CatBoost: {catboost_params}")
        except Exception as e:
            log_error(f"Error al definir parámetros: {e}")
            return

        try:
            # Construir pipeline
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=english_stopwords, max_features=100, lowercase=True, token_pattern=r'\b\w+\b')),
                ('catboost', CatBoostClassifier(**catboost_params))
            ])
            logging.info("Pipeline construido exitosamente.")
        except Exception as e:
            log_error(f"Error al construir el pipeline: {e}")
            return

        # Verificar si hay modelo campeón
        try:
            from NB_funciones import get_champion_model, compare_models
            champion_model, _ = get_champion_model(model_name=CONFIG["model"]["name"])
        except mlflow.exceptions.MlflowException:
            log_error("Fallo en la conexión con MLflow mientras obtenía el modelo campeón.")
            return

        ''' 
        try:
            # Solo entrenar si es necesario
            if not champion_model:
                log_info("No hay modelo campeón, por lo que no se realizará comparación.")
                return

            champion_metrics = evaluate_model(champion_model, data_splits["X_test"], data_splits["y_test"])

            temp_model = pipeline  # Usa el pipeline en lugar del modelo directamente
            temp_model.fit(data_splits["X_train"][text_column], data_splits["y_train"])

            # Evaluar challenger solo si hay un modelo campeón
            challenger_metrics = evaluate_model(temp_model, data_splits["X_test"][text_column], data_splits["y_test"])

            is_better, all_metrics, comparison = compare_models(champion_metrics, challenger_metrics, "accuracy", CONFIG["model"]["champion_threshold"])

            if not is_better:
                log_info("El modelo retador no supera al campeón, terminando ejecución.")
                return

            log_info("El modelo retador supera al campeón, continuando con el entrenamiento.")

        except Exception as e:
            log_error(f"Error al comparar modelos: {e}")
            return

        '''

        try:
        # Registrar modelo solo si supera al campeón
            with mlflow.start_run(experiment_id=experiment_id):

                try:
                    # Inferir signature para input/output
                    signature = infer_signature(data_splits["X_train"], data_splits["y_train"])
                    log_info(f"Signature inferida: {signature.inputs}, {signature.outputs}")
                except Exception as e:
                    log_error(f"Error al inferir signature: {e}")
                    return



                try:
                    # Registra el modelo con un ejemplo de entrada
                    input_example = np.array(data_splits["X_train"][:1])  # Toma una muestra como ejemplo de entrada
                    log_info(f"Ejemplo de entrada: {input_example}")
                except Exception as e:
                    log_error(f"Error al registrar el ejemplo de entrada: {e}")
                    return


                try:
                    model = pipeline  # Usa el pipeline en lugar del modelo directamente
                    model.fit(data_splits["X_train"], data_splits["y_train"]) 
                    mlflow.sklearn.log_model(model, CONFIG["model"]["name"], signature=signature, input_example=input_example) # signature=signature, input_example=input_example
                    log_info(f"Modelo registrado: {CONFIG['model']['name']}")
                except Exception as e:
                    log_error(f"Error al registrar el modelo: {e}")
                    return

                try:
                    # Registra las métricas
                    accuracy = pipeline.score(data_splits["X_test"], data_splits["y_test"])
                    mlflow.log_metric("accuracy", accuracy)

                    # Registra los hiperparámetros del modelo
                    mlflow.log_params(catboost_params)
                    log_info(f"Parámetros registrados: {catboost_params}")
                except Exception as e:
                    log_error(f"Error al registrar las métricas y parámetros: {e}")
                    return

                log_info(f"Modelo registrado con ID: {mlflow.active_run().info.run_id}")
                logging.info(f"Modelo registrado con ID: {mlflow.active_run().info.run_id}")
        except Exception as e:
            log_error(f"Error al registrar el modelo: {e}")
            return

        log_info("Pipeline completado exitosamente.")

    except Exception as e:
        log_error(f"Error crítico en el pipeline: {e}")
        raise e



# In[6]:


if __name__ == "__main__":
    main()


# In[ ]:


#!jupyter nbconvert --to script NB_entrenamiento_final.ipynb

