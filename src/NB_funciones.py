#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from dotenv import load_dotenv
import os

import unicodedata
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score, f1_score, fbeta_score, roc_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import logging
import json
import re
import string 
import joblib
import warnings
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from mlflow.models.signature import infer_signature
from datetime import datetime

warnings.filterwarnings('ignore')


# In[2]:


# Funciones para registrar mensajes tanto en log como en consola
def log_info(message):
    """
    Registra y muestra un mensaje de información.

    Esta función imprime el mensaje proporcionado en la consola y lo registra 
    utilizando el nivel de información ('INFO') del módulo 'logging'. Es útil 
    para el seguimiento de eventos importantes sin afectar el flujo de ejecución.

    Args:
        message (str): El mensaje de información a registrar y mostrar.

    Returns:
        None
    """
    print(message)
    logging.info(message)

def log_error(message):
    """
    Registra y muestra un mensaje de error.

    Esta función imprime el mensaje de error en la consola con una etiqueta `ERROR:`
    para una identificación clara y lo registra en el sistema de logs con nivel `ERROR`, 
    lo que ayuda en el monitoreo y depuración de fallos en la ejecución del programa.

    Args:
        message (str): El mensaje descriptivo del error que se desea registrar.

    Returns:
        None
    """
    print(f'ERROR: {message}')
    logging.error(message)


# In[3]:



# Configuración de logging
logging.basicConfig(level=logging.INFO)
log_info = logging.info
log_error = logging.error

class CargarDatos:
    """
    Clase para cargar datos desde un archivo CSV cuya ruta está almacenada en un archivo .env.

    Esta clase ayuda a gestionar la carga de datos de manera segura y modular, 
    obteniendo la ruta del archivo CSV desde una variable de entorno definida 
    en un archivo '.env'. Además, maneja errores de acceso y proporciona registros informativos.
    """

    def __init__(self, clave_env):
        """
        Inicializa la clase cargando la variable de entorno y validando la ruta del archivo CSV.

        Args:
            clave_env (str): Nombre de la clave en el archivo .env que almacena la ruta del archivo CSV.
        
        Raises:
            ValueError: Si la clave proporcionada no está definida en el archivo .env.
        """

        # Cargar las variables de entorno
        load_dotenv()
        
        # Obtener la ruta desde el archivo .env
        self.ruta_archivo = os.getenv(clave_env)
        if not self.ruta_archivo:
            raise ValueError(f"La clave '{clave_env}' no está definida en .env")
        
        self.df = None

    def cargar_csv(self):
        """
        Carga los datos desde el archivo CSV y los almacena en un DataFrame.

        La función intenta leer el archivo CSV usando `pandas`, registra un mensaje informativo 
        si la carga es exitosa y maneja posibles errores de lectura.

        Returns:
            pd.DataFrame: DataFrame con los datos cargados desde el archivo CSV.

        Raises:
            Exception: Propaga cualquier excepción ocurrida durante la carga del archivo.
        """

        try:
            self.df = pd.read_csv(self.ruta_archivo)
            log_info(f"Archivo {self.ruta_archivo} cargado correctamente.")
        except Exception as e:
            log_error(f"Error al cargar el archivo {self.ruta_archivo}: {e}")
            raise e
        
        return self.df



# In[4]:


class TextPreprocessor:
    '''
    Clase para preprocesar texto.
    Atributos:
        stop_words (set): Conjunto de palabras vacías en ingles.
        nlp (spacy.lang.en.EN): Modelo de lenguaje en ingles de spaCy.
    Métodos:
        __init__(): Inicializa el preprocesador de texto.
        limpiar_texto(): Limpia el texto eliminando caracteres no deseados.
        tokenizar_texto(): Tokeniza el texto y elimina palabras vacías.
        lematizar_texto(): Lematiza el texto utilizando spaCy.
        preprocesar_texto(): Preprocesa el texto completo.
    '''
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.nlp = spacy.load("en_core_web_sm")

    def limpiar_texto(self, texto):
        # Eliminar caracteres no deseados
        try:
            texto = texto.lower()
            texto = re.sub(r'\d+', '', texto)  # Eliminar números
            texto = re.sub(r'\s+', ' ', texto)  # Eliminar espacios extra
            texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)  # Eliminar URLs
            texto = re.sub(r'@\w+', '', texto)  # Eliminar menciones
            texto = re.sub(r'#', '', texto)  # Eliminar hashtags
            texto = re.sub(r'[^a-zA-Z0-9áéíóúüñÑÁÉÍÓÚÜ\s]', '', texto)
            log_info('Texto limpiado correctamente.')
        except Exception as e:
            log_error(f'Error al limpiar el texto: {e}')
            raise e
        return texto

    def tokenizar_texto(self, texto):
        # Tokenizar y eliminar palabras vacías
        try:
            tokens = word_tokenize(texto.lower())
            tokens = [token for token in tokens if token not in self.stop_words]
            log_info('Tokens generados correctamente.')
        except Exception as e:
            log_error(f'Error al tokenizar el texto: {e}')
            raise e
        return tokens

    def lematizar_texto(self, tokens):
        # Lematizar los tokens
        try:
            doc = self.nlp(' '.join(tokens))
            lemas = [token.lemma_ for token in doc]
            lemas = [lema for lema in lemas if lema not in self.stop_words]
            lemas = [lema for lema in lemas if len(lema) > 1]  # Eliminar lemas de longitud 1
            lemas = [lema for lema in lemas if not re.match(r'^[a-zA-Z0-9]+$', lema)]  # Eliminar lemas que son solo números
            log_info('Lematización completada correctamente.')
        except Exception as e:
            log_error(f'Error al lematizar el texto: {e}')
            raise e
        return lemas

    def preprocesar_texto(self, texto):
        # Preprocesar el texto completo
        try:
            texto_limpio = self.limpiar_texto(texto)
            tokens = self.tokenizar_texto(texto_limpio)
            lemas = self.lematizar_texto(tokens)
            log_info('Texto preprocesado correctamente.')
        except Exception as e:
            log_error(f'Error al preprocesar el texto: {e}')
            raise e
        return ' '.join(lemas)


# In[5]:


def run_ml_experiment(X_train, y_train, X_test, y_test, english_stopwords, model_name="modelo_catboost_prueba"):
    """
    Ejecuta un experimento de machine learning con CatBoost y MLflow.

    Parámetros:
    - X_train, y_train: Datos de entrenamiento.
    - X_test, y_test: Datos de prueba.
    - english_stopwords: Lista de palabras irrelevantes para la vectorización.
    - model_name: Nombre para registrar el modelo en MLflow.

    Registra el modelo, métricas y parámetros en MLflow.
    """
    try:
        print("Iniciando el experimento...")

        # Definir los hiperparámetros del modelo
        catboost_params = {
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.1,
            "verbose": 100
        }

        # Construir el pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words=english_stopwords, max_features=100, lowercase=True, token_pattern=r'\b\w+\b')),
            ('catboost', CatBoostClassifier(**catboost_params))
        ])

        with mlflow.start_run():
            # Entrenar el modelo
            pipeline.fit(X_train, y_train)

            # Inferir la firma del modelo
            signature = infer_signature(X_train, y_train)

            # Registrar el modelo en MLflow
            input_example = np.array(X_train[:1])  
            mlflow.sklearn.log_model(pipeline, "modelo_catboost", 
                                     input_example=input_example, 
                                     signature=signature, 
                                     registered_model_name=model_name)

            # Registrar métricas y parámetros
            accuracy = pipeline.score(X_test, y_test)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_params(catboost_params)

            print(f"Modelo registrado con precisión: {accuracy}")
            logging.info(f"Modelo registrado con precisión: {accuracy}")

    except Exception as e:
        logging.error(f"Error durante la ejecución de MLflow: {e}")
        raise e


# In[6]:


def plot_confusion_matrix(model, X, y, title='Matriz de Confusión'):
    """
    Grafica la matriz de confusión para un modelo de clasificación.

    Esta función genera una representación visual de la matriz de confusión utilizando 
    'ConfusionMatrixDisplay.from_estimator'. Es útil para evaluar el rendimiento del modelo 
    y entender la distribución de predicciones correctas e incorrectas.

    Args:
        model: Modelo entrenado que se utilizará para predecir etiquetas.
        X: Datos de entrada para generar la matriz de confusión.
        y: Etiquetas verdaderas correspondientes a los datos de entrada.
        title (str, opcional): Título del gráfico. Por defecto es 'Matriz de Confusión'.

    Returns:
    matplotlib.figure.Figure: Figura de la matriz de confusión generada.

    Raises:
        Exception: Captura errores durante la generación del gráfico y los registra en el sistema de logs.
    """
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_estimator(model, X, y, ax=ax)
        plt.title(title)
        plt.tight_layout()
        return fig
    except Exception as e:
        log_error(f'Error al graficar matriz de confusión: {e}')
        return None


# In[7]:


def plot_roc_curve(model, X, y, model_name='Modelo'):
    """
    Grafica la curva ROC (Receiver Operating Characteristic) para evaluar el rendimiento de un modelo de clasificación binaria.

    La curva ROC muestra la relación entre la tasa de verdaderos positivos ('True Positive Rate') y la tasa de falsos positivos ('False Positive Rate'),
    permitiendo evaluar la capacidad del modelo para discriminar entre clases. El área bajo la curva (AUC) es una métrica útil para medir la calidad del modelo.

    Args:
        model: Modelo de clasificación entrenado que debe soportar 'predict_proba'.
        X: Datos de entrada utilizados para generar las predicciones.
        y: Etiquetas verdaderas correspondientes a los datos de entrada.
        model_name (str, opcional): Nombre del modelo para incluir en la leyenda del gráfico. Por defecto es 'Modelo'.

    Returns:
        matplotlib.figure.Figure | None: Figura de la curva ROC si la ejecución es exitosa, 'None' en caso de error.

    Raises:
        Exception: Captura errores en la generación del gráfico y los registra en el sistema de logs.
    """
    try:
        if not hasattr(model, 'predict_proba'):
            log_info(f'El modelo {model_name} no soporta predict_proba, no se puede graficar ROC.')
            return None

        y_pred_proba = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Curva ROC')
        ax.legend(loc='lower right')
        return fig
    except Exception as e:
        log_error(f'Error al graficar curva ROC: {e}')
        return None


# In[8]:


def get_champion_model(model_name='Modelo'):
    '''
    Recupera el modelo campeón actual desde MLflow.

    Esta función busca la última versión del modelo especificado en MLflow, 
    comenzando por el entorno de producción (`Production`). Si no encuentra una 
    versión en producción, intenta recuperar una versión en `Staging`. 
    Si aún no hay versiones disponibles, toma la versión más reciente del modelo 
    registrado en MLflow. Finalmente, carga el modelo y su `run_id` correspondiente.

    Args:
        model_name (str, opcional): Nombre del modelo registrado en MLflow. 
                                    Por defecto es 'Modelo'.

    Returns:
        tuple: 
            - Modelo cargado desde MLflow (`sklearn model`).
            - Identificador de ejecución (`run_id`).
            - Retorna `(None, None)` si no hay modelos disponibles.

    Raises:
        Exception: Maneja posibles errores durante la recuperación del modelo.
    '''
    try:
        client = mlflow.tracking.MlflowClient()

        # Buscar la última versión del modelo campeón
        try:
            latest_version = client.get_latest_versions(model_name, stages=['Production'])
            if not latest_version:
                log_info(f'No se encontró un modelo {model_name} en producción. Buscando en staging...')
                latest_version = client.get_latest_versions(model_name, stages=['Staging'])

            if not latest_version:
                log_info(f'No se encontró un modelo {model_name} en staging. Buscando la versión más reciente...')
                latest_version = client.get_latest_versions(model_name)

            if latest_version:
                model_uri = f'models:/{model_name}/{latest_version[0].version}'
                champion_model = mlflow.sklearn.load_model(model_uri)
                log_info(f'Modelo campeón cargado: {model_name} version {latest_version[0].version}')
                return champion_model, latest_version[0].run_id
            else:
                log_info(f'No se encontró ningún modelo registrado con el nombre {model_name}')
                return None, None
        except Exception as e:
            log_error(f'No se pudo obtener la última versión del modelo: {e}')
            return None, None

    except Exception as e:
        log_error(f'Error al recuperar el modelo campeón: {e}')
        return None, None


# In[9]:


def register_challenger_model(model, metrics, X_train, y_train, is_champion=False, CHAMPION_MODEL_NAME='Modelo_Campeon'):
    '''
    Registra un modelo desafiante o campeón en MLflow.

    Esta función guarda el modelo en MLflow con su firma y métricas, asignándole una etapa (`Staging` o `Production`).
    Si el modelo se registra como campeón (`is_champion=True`), se promueve directamente a producción.

    Args:
        model: Modelo entrenado que se desea registrar en MLflow.
        metrics (dict): Diccionario de métricas a registrar (ejemplo: {'accuracy': 0.92}).
        X_train: Datos de entrenamiento usados para inferir la firma del modelo.
        y_train: Etiquetas de entrenamiento correspondientes a X_train.
        is_champion (bool, opcional): Indica si el modelo es el campeón. Por defecto es `False`.
        CHAMPION_MODEL_NAME (str, opcional): Nombre base para los modelos. Por defecto es 'Modelo_Campeon'.

    Returns:
        bool: `True` si el registro es exitoso, `False` en caso de error.

    Raises:
        Exception: Maneja cualquier error durante el registro y lo registra en los logs.
    '''
    try:
        # Inferir firma para input/output
        signature = infer_signature(X_train, y_train)

        # Registrar el modelo con un ejemplo de entrada
        input_example = np.array(X_train[:1])

        # Nombre del modelo y etapa
        model_name = CHAMPION_MODEL_NAME if is_champion else f'{CHAMPION_MODEL_NAME}_challenger'
        stage = 'Production' if is_champion else 'Staging'

        # Registrar modelo
        mlflow.sklearn.log_model(
            model, 
            'model', 
            input_example=input_example, 
            signature=signature, 
            registered_model_name=model_name
        )

        # Registrar métricas
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Si es el campeón, mover a producción
        if is_champion:
            client = mlflow.tracking.MlflowClient()
            latest_version = client.get_latest_versions(model_name)[0].version
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version,
                stage=stage
            )
            log_info(f'Modelo {model_name} v{latest_version} promocionado a {stage}')

        return True
    except Exception as e:
        log_error(f'Error al registrar el modelo: {e}')
        return False


# In[10]:


def train_challenger_model(X_train, y_train, X_valid=None, y_valid=None, hyperparams=None):
    '''
    Entrena un modelo desafiante utilizando CatBoost y TfidfVectorizer.

    Este modelo se construye dentro de un pipeline de `scikit-learn`, combinando 
    `TfidfVectorizer` para transformar texto y `CatBoostClassifier` como clasificador. 
    Se permite el uso de un conjunto de validación opcional para aplicar `early stopping`.

    Args:
        X_train: Datos de entrenamiento (entrada del modelo).
        y_train: Etiquetas correspondientes a `X_train`.
        X_valid (opcional): Datos de validación para mejorar el ajuste del modelo.
        y_valid (opcional): Etiquetas de validación correspondientes a `X_valid`.
        hyperparams (dict, opcional): Diccionario de hiperparámetros para `CatBoostClassifier`. 
                                      Si no se proporcionan, se utilizan valores por defecto.

    Returns:
        sklearn.pipeline.Pipeline: Pipeline entrenado con `TfidfVectorizer` y `CatBoostClassifier`.

    Raises:
        Exception: Captura errores durante el entrenamiento y los registra en los logs.
    '''
    try:
        # Parámetros por defecto si no se especifican
        if hyperparams is None:
            hyperparams = {
                'iterations': 500,
                'depth': 6,
                'learning_rate': 0.1,
                'verbose': 100,
                'max_features': 100
            }

        # Extraer parámetros específicos de vectorizador y modelo
        max_features = hyperparams.pop('max_features', 100)

        log_info(f'Entrenando modelo desafiante con parámetros: {hyperparams}')

        english_stopwords = stopwords.words('english')

        # Crear pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words=english_stopwords, max_features=max_features, lowercase=True, token_pattern=r'\b\w+\b')),              
            ('catboost', CatBoostClassifier(**hyperparams))
        ])

        # Entrenar modelo
        if X_valid is not None and y_valid is not None:
            # Usar conjunto de validación para early stopping
            pipeline.fit(X_train, y_train, catboost__eval_set=[(X_valid, y_valid)])
        else:
            pipeline.fit(X_train, y_train)

        log_info('Modelo desafiante entrenado correctamente')
        return pipeline
    except Exception as e:
        log_error(f'Error al entrenar modelo desafiante: {e}')
        raise e


# In[11]:


def compare_models(champion_metrics, challenger_metrics, primary_metric="accuracy", threshold=0.5):
    '''
    Compara dos modelos y determina si el desafiante debe reemplazar al campeón.

    Esta función evalúa las métricas de rendimiento de un modelo campeón y un modelo desafiante.
    Se compara la métrica primaria especificada (`primary_metric`) y todas las métricas disponibles
    para determinar si el desafiante supera al campeón según un umbral (`threshold`).

    Args:
        champion_metrics (dict | None): Métricas del modelo campeón. Si es `None`, el desafiante se convierte en campeón por defecto.
        challenger_metrics (dict): Métricas del modelo desafiante.
        primary_metric (str, opcional): Métrica clave para la comparación. Por defecto es `"accuracy"`.
        threshold (float, opcional): Umbral mínimo de mejora requerido para que el desafiante reemplace al campeón. Por defecto es `0.5`.

    Returns:
        tuple:
            - bool: `True` si el desafiante debe reemplazar al campeón, `False` en caso contrario.
            - dict: Comparación detallada de todas las métricas disponibles.
            - dict: Resumen de comparación de la métrica primaria (`champion`, `challenger`, `absolute_diff`, `percent_diff`).

    Raises:
        Exception: Captura errores en la evaluación y los registra en los logs.
    '''
    try:
        if champion_metrics is None:
            log_info('No hay modelo campeón para comparar. El desafiante se convierte en campeón por defecto.')
            return True, {}, {'champion': None, 'challenger': challenger_metrics.get(primary_metric, None)}
        
        if not challenger_metrics:
            log_info('No hay modelo desafiante disponible. Se mantiene el modelo campeón.')
            return False, {}, {}

        # Comparar métricas primarias
        champion_score = champion_metrics[primary_metric]
        challenger_score = challenger_metrics[primary_metric]

        improvement = challenger_score - champion_score
        percent_improvement = (improvement / champion_score) * 100 if champion_score > 0 else float('inf')

        comparison = {
            'champion': champion_score,
            'challenger': challenger_score,
            'absolute_diff': improvement,
            'percent_diff': percent_improvement
        }

        # Comparar todas las métricas disponibles
        all_metrics = {}
        for metric in set(champion_metrics.keys()).union(challenger_metrics.keys()):
            if metric in champion_metrics and metric in challenger_metrics:
                champion_val = champion_metrics[metric]
                challenger_val = challenger_metrics[metric]
                diff = challenger_val - champion_val
                perc_diff = (diff / champion_val) * 100 if champion_val > 0 else float('inf')

                all_metrics[metric] = {
                    'champion': champion_val,
                    'challenger': challenger_val,
                    'absolute_diff': diff,
                    'percent_diff': perc_diff
                }

        # Determinar si el desafiante es mejor
        is_better = improvement > threshold

        if is_better:
            log_info(f'El modelo desafiante es mejor en {primary_metric}: {challenger_score:.4f} vs {champion_score:.4f} ')
            log_info(f'Mejora absoluta: {improvement:.4f}, Mejora porcentual: {percent_improvement:.2f}%')
        else:
            log_info(f'El modelo desafiante NO supera al campeón en {primary_metric}: {challenger_score:.4f} vs {champion_score:.4f}')
            log_info(f'Diferencia absoluta: {improvement:.4f}, Diferencia porcentual: {percent_improvement:.2f}%')

        return is_better, all_metrics, comparison
    except Exception as e:
        log_error(f'Error al comparar modelos: {e}')
        return False, {}, {}


# In[12]:
def setup_environment(log_file="training.log"):
    """
    Configura el entorno de ejecución ajustando la gestión de warnings, el sistema de logging y la descarga de recursos NLTK.

    Esta función desactiva ciertos warnings, configura un sistema de logging para registrar eventos y 
    descarga los recursos necesarios de NLTK para el procesamiento de texto.

    Args:
        log_file (str, opcional): Nombre del archivo donde se almacenarán los logs. Por defecto, "training.log".

    Returns:
        list: Lista de palabras de detención en inglés proporcionada por NLTK.

    Raises:
        Exception: Captura errores durante la descarga de recursos NLTK y los registra en los logs.
    """
    warnings.filterwarnings('ignore')
    
    # Configuración de logging
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Descargar recursos NLTK necesarios
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        log_info("Recursos NLTK descargados correctamente")
    except Exception as e:
        log_error(f"Error descargando recursos NLTK: {e}")
        
    return stopwords.words('english')

# In[13]:

def preprocess_data(data, text_column, target_column, test_size=0.3, valid_size=0.3, random_state=42):
    """
    Preprocesa los datos y divide en conjuntos de entrenamiento, validación y prueba.

    La función convierte la columna objetivo (`target_column`) a tipo entero y realiza una 
    división estratificada para preservar la distribución de clases. Primero se separa el 
    conjunto de prueba (`test`), y posteriormente se divide el conjunto de entrenamiento 
    (`train`) para obtener un subconjunto de validación (`valid`).

    Args:
        data (pd.DataFrame): DataFrame con los datos originales.
        text_column (str): Nombre de la columna que contiene los datos de entrada (texto).
        target_column (str): Nombre de la columna con la variable objetivo.
        test_size (float, opcional): Proporción de datos destinada al conjunto de prueba. Por defecto es `0.3`.
        valid_size (float, opcional): Proporción de datos del entrenamiento destinada a validación. Por defecto es `0.3`.
        random_state (int, opcional): Valor de aleatoriedad para reproducibilidad. Por defecto es `42`.

    Returns:
        dict: Diccionario con los conjuntos de datos divididos (`X_train`, `y_train`, `X_valid`, `y_valid`, `X_test`, `y_test`).

    Raises:
        Exception: Captura errores en el preprocesamiento y los registra en los logs.
    """
    try:
        # Asegurar que el target sea int
        data[target_column] = data[target_column].astype('int')
        
        # División inicial train/test
        X_train, X_test, y_train, y_test = train_test_split(
            data[text_column], 
            data[target_column], 
            test_size=test_size, 
            stratify=data[target_column], 
            random_state=random_state
        )
        
        # División adicional para conjunto de validación
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, 
            y_train, 
            test_size=valid_size, 
            stratify=y_train, 
            random_state=random_state
        )
        
        log_info(f'Datos divididos: Train {X_train.shape[0]}, Valid {X_valid.shape[0]}, Test {X_test.shape[0]} registros')
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_valid': X_valid, 'y_valid': y_valid,
            'X_test': X_test, 'y_test': y_test
        }
    except Exception as e:
        log_error(f'Error en el preprocesamiento de datos: {e}')
        raise e
    
# In[14]:
def create_mlflow_experiment(experiment_name):
    """
    Crea o configura un experimento en MLflow.

    Esta función intenta crear un nuevo experimento en MLflow con el nombre especificado.
    Si el experimento ya existe, en lugar de generarlo nuevamente, lo establece como el 
    experimento activo para futuras ejecuciones.

    Args:
        experiment_name (str): Nombre del experimento en MLflow.

    Returns:
        None

    Raises:
        Exception: Captura errores en la creación del experimento y los registra en los logs.
    """
    try:
        mlflow.create_experiment(experiment_name)
        log_info(f"Experimento '{experiment_name}' creado exitosamente")
    except:
        mlflow.set_experiment(experiment_name)
        log_info(f"Usando experimento existente '{experiment_name}'")

# In[15]:
def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evalúa un modelo de clasificación y devuelve métricas de rendimiento detalladas.

    La función calcula métricas de evaluación básicas, como `accuracy`, `precision`, `recall` y `f1_score`.
    Si el modelo soporta `predict_proba`, también calcula métricas avanzadas como `ROC AUC` y `average precision`.

    Args:
        model: Modelo de clasificación entrenado.
        X_test: Datos de entrada del conjunto de prueba.
        y_test: Etiquetas verdaderas correspondientes a `X_test`.
        threshold (float, opcional): Umbral para clasificación basado en probabilidades. Actualmente no se usa en la función.

    Returns:
        dict: Diccionario con métricas calculadas (`accuracy`, `precision`, `recall`, `f1_score`, `roc_auc`, `average_precision`).

    Raises:
        Exception: Captura errores en la evaluación del modelo y los registra en los logs.
    """
    try:
        # Predicciones
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Métricas básicas
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1_score": f1_score(y_test, y_pred, average='weighted'),
        }
        
        # Métricas avanzadas si disponemos de probabilidades
        if y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
            metrics["average_precision"] = average_precision_score(y_test, y_proba)
        
        log_info(f"Evaluación completa del modelo - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
        return metrics
    
    except Exception as e:
        log_error(f"Error evaluando el modelo: {e}")
        raise e
    
# In[16]:
def save_model_artifacts(model, model_name, version, metrics, data_signature=None):
    """
    Guarda los artefactos del modelo, incluyendo métricas, el modelo serializado y su firma de datos.

    Esta función crea un directorio de salida donde almacena información relevante sobre el modelo 
    entrenado, como sus métricas en formato JSON, su representación serializada con `joblib` y su 
    firma de datos si está disponible.

    Args:
        model: Modelo entrenado que se desea guardar.
        model_name (str): Nombre del modelo.
        version (str | int): Versión del modelo que se está guardando.
        metrics (dict): Diccionario con las métricas del modelo.
        data_signature (dict, opcional): Firma del modelo para describir la estructura de entrada. Por defecto es `None`.

    Returns:
        str | None: Ruta del directorio donde se guardaron los artefactos, o `None` si ocurre un error.

    Raises:
        Exception: Captura errores en el proceso de guardado y los registra en los logs.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"model_artifacts/{model_name}/{version}_{timestamp}"
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar métricas en JSON
        with open(f"{output_dir}/metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
            
        # Guardar modelo usando joblib (como backup adicional a MLflow)
        joblib.dump(model, f"{output_dir}/{model_name}_{version}.joblib")
        
        # Guardar firma del modelo si está disponible
        if data_signature:
            with open(f"{output_dir}/data_signature.json", 'w') as f:
                json.dump(data_signature, f, indent=2)
        
        log_info(f"Artefactos del modelo guardados en {output_dir}")
        return output_dir
    except Exception as e:
        log_error(f"Error guardando artefactos del modelo: {e}")
        return None
    
# In[17]:
def generate_model_card(model_info, metrics, output_path):
    """
    Genera una tarjeta de modelo (`model card`) con información clave sobre un modelo de machine learning.

    Una model card proporciona documentación sobre el modelo, incluyendo su rendimiento, datos de entrenamiento, 
    parámetros utilizados, limitaciones y casos de uso previstos. La función guarda esta información en formato JSON.

    Args:
        model_info (dict): Diccionario con información del modelo, incluyendo:
            - `"name"` (str): Nombre del modelo.
            - `"version"` (str | int): Versión del modelo.
            - `"description"` (str, opcional): Descripción general del modelo.
            - `"training_data"` (dict, opcional): Información sobre los datos de entrenamiento.
            - `"parameters"` (dict, opcional): Hiperparámetros utilizados en el modelo.
            - `"limitations"` (str, opcional): Restricciones o consideraciones del modelo.
            - `"intended_use"` (str, opcional): Propósito y contexto de uso del modelo.
        metrics (dict): Diccionario con las métricas de rendimiento del modelo.
        output_path (str): Ruta donde se guardará la model card en formato JSON.

    Returns:
        bool: `True` si la model card se genera correctamente, `False` en caso de error.

    Raises:
        Exception: Captura errores en la creación de la model card y los registra en los logs.
    """
    try:
        model_card = {
            "model_name": model_info["name"],
            "version": model_info["version"],
            "description": model_info.get("description", ""),
            "created_at": datetime.now().isoformat(),
            "performance_metrics": metrics,
            "training_data": model_info.get("training_data", {}),
            "parameters": model_info.get("parameters", {}),
            "limitations": model_info.get("limitations", ""),
            "intended_use": model_info.get("intended_use", ""),
        }
        
        with open(output_path, 'w') as f:
            json.dump(model_card, f, indent=2)
            
        log_info(f"Model card generada en {output_path}")
        return True
    except Exception as e:
        log_error(f"Error generando model card: {e}")
        return False
    
# In[18]:
def data_drift_detection(reference_data, current_data, columns=None, threshold=0.1):
    """
    Detecta data drift entre dos conjuntos de datos
    
    Args:
        reference_data: DataFrame con datos de referencia (entrenamiento original)
        current_data: DataFrame con datos actuales
        columns: Lista de columnas a analizar (None = todas)
        threshold: Umbral para considerar drift significativo
        
    Returns:
        Dict con resultados de análisis de drift
    """
    try:
        if columns is None:
            # Si no se especifican columnas, usar todas las comunes entre ambos dataframes
            columns = [col for col in reference_data.columns if col in current_data.columns]
        
        results = {
            "drift_detected": False,
            "details": {}
        }
        
        for col in columns:
            # Simplificación: usando estadísticas básicas para detectar drift
            # En una implementación real, usar métodos como KS test o PSI (Population Stability Index)
            if pd.api.types.is_numeric_dtype(reference_data[col]):
                ref_mean = reference_data[col].mean()
                cur_mean = current_data[col].mean()
                
                ref_std = reference_data[col].std()
                cur_std = current_data[col].std()
                
                # Calcular cambio relativo
                mean_change = abs(ref_mean - cur_mean) / (abs(ref_mean) if abs(ref_mean) > 0 else 1)
                std_change = abs(ref_std - cur_std) / (abs(ref_std) if abs(ref_std) > 0 else 1)
                
                col_drift = mean_change > threshold or std_change > threshold
                
                results["details"][col] = {
                    "drift_detected": col_drift,
                    "metrics": {
                        "mean_change": mean_change,
                        "std_change": std_change
                    }
                }
                
                if col_drift:
                    results["drift_detected"] = True
            
            elif pd.api.types.is_categorical_dtype(reference_data[col]) or pd.api.types.is_object_dtype(reference_data[col]):
                # Para variables categóricas, comparar distribución de valores
                ref_dist = reference_data[col].value_counts(normalize=True).to_dict()
                cur_dist = current_data[col].value_counts(normalize=True).to_dict()
                
                # Calcular diferencia en distribuciones
                all_categories = set(ref_dist.keys()) | set(cur_dist.keys())
                max_diff = 0
                
                for cat in all_categories:
                    ref_val = ref_dist.get(cat, 0)
                    cur_val = cur_dist.get(cat, 0)
                    diff = abs(ref_val - cur_val)
                    max_diff = max(max_diff, diff)
                
                col_drift = max_diff > threshold
                
                results["details"][col] = {
                    "drift_detected": col_drift,
                    "metrics": {
                        "max_category_difference": max_diff
                    }
                }
                
                if col_drift:
                    results["drift_detected"] = True
        
        log_info(f"Análisis de data drift completado. Drift detectado: {results['drift_detected']}")
        return results
    
    except Exception as e:
        log_error(f"Error en detección de data drift: {e}")
        raise e
    
# In[19]:
def model_performance_monitoring(model, data, target, prediction_column=None, alert_threshold=0.1):
    """
    Monitorea el rendimiento del modelo en datos nuevos y detecta degradación
    
    Args:
        model: Modelo entrenado
        data: DataFrame con datos nuevos
        target: Nombre de columna target o Series con valores target
        prediction_column: Nombre de columna con predicciones (si ya existen en el DataFrame)
        alert_threshold: Umbral para alerta de degradación de rendimiento
        
    Returns:
        Dict con métricas y alertas
    """
    try:
        # Preparar datos
        X = data.drop(columns=[target]) if isinstance(target, str) and target in data.columns else data
        y_true = data[target] if isinstance(target, str) and target in data.columns else target
        
        # Obtener predicciones
        if prediction_column and prediction_column in data.columns:
            y_pred = data[prediction_column]
        else:
            y_pred = model.predict(X)
        
        # Calcular métricas
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred, average='weighted'),
            "precision": precision_score(y_true, y_pred, average='weighted'),
            "recall": recall_score(y_true, y_pred, average='weighted')
        }
        
        # Comparar con métricas de referencia del modelo (si disponibles)
        reference_metrics = {}
        if hasattr(model, 'metadata') and 'reference_metrics' in model.metadata:
            reference_metrics = model.metadata['reference_metrics']
        
        # Detectar degradación
        alerts = []
        if reference_metrics:
            for metric_name, current_value in metrics.items():
                if metric_name in reference_metrics:
                    ref_value = reference_metrics[metric_name]
                    degradation = ref_value - current_value
                    relative_degradation = degradation / ref_value if ref_value > 0 else 0
                    
                    if relative_degradation > alert_threshold:
                        alerts.append({
                            "metric": metric_name,
                            "reference": ref_value,
                            "current": current_value,
                            "degradation": relative_degradation
                        })
        
        result = {
            "metrics": metrics,
            "alerts": alerts,
            "degradation_detected": len(alerts) > 0
        }
        
        if result["degradation_detected"]:
            log_info(f"Degradación de rendimiento detectada: {alerts}")
        else:
            log_info("Monitoreo de rendimiento completado sin alertas")
            
        return result
    
    except Exception as e:
        log_error(f"Error en monitoreo de rendimiento: {e}")
        raise e
    
# In[20]:
def run_model_training_pipeline(
    data, text_column, target_column, experiment_name, model_name, version="1.0.0",
    test_size=0.3, valid_size=0.3, random_state=42, optimization_metric="f1_score",
    champion_threshold=0.5
):
    """
    Ejecuta el pipeline completo de entrenamiento de un modelo de clasificación de texto.

    Este pipeline abarca la configuración del entorno, la preparación de datos, el entrenamiento del modelo,
    la evaluación, el registro en MLflow y la comparación con un modelo campeón si existe.

    Args:
        data (pd.DataFrame): DataFrame con los datos de entrenamiento.
        text_column (str): Nombre de la columna que contiene los textos a clasificar.
        target_column (str): Nombre de la columna con la variable objetivo.
        experiment_name (str): Nombre del experimento en MLflow.
        model_name (str): Nombre del modelo a registrar.
        version (str, opcional): Versión del modelo. Por defecto `"1.0.0"`.
        test_size (float, opcional): Proporción de datos destinada al conjunto de prueba. Por defecto `0.3`.
        valid_size (float, opcional): Proporción del conjunto de entrenamiento destinada a validación. Por defecto `0.3`.
        random_state (int, opcional): Valor de aleatoriedad para reproducibilidad. Por defecto `42`.
        optimization_metric (str, opcional): Métrica clave para la comparación de modelos. Por defecto `"f1_score"`.
        champion_threshold (float, opcional): Umbral mínimo de mejora para reemplazar el modelo campeón. Por defecto `0.5`.

    Returns:
        tuple:
            - Modelo entrenado (`sklearn model`).
            - Diccionario con métricas de evaluación.
            - Identificador de ejecución en MLflow (`run_id`).

    Raises:
        Exception: Captura errores en el pipeline y los registra en los logs.
    """
    try:
        # 1. Configurar entorno
        stopwords_list = setup_environment()
        
        # 2. Preparar datos
        data_splits = preprocess_data(
            data, text_column, target_column, 
            test_size=test_size, valid_size=valid_size, 
            random_state=random_state
        )
        
        # 3. Configurar experimento MLflow
        create_mlflow_experiment(experiment_name)
        
        # 4. Entrenar modelo y registrar experimento
        with mlflow.start_run(run_name=f"{model_name}_v{version}") as run:
            # Ejecutar experimento existente
            model = run_ml_experiment(
                data_splits['X_train'], data_splits['y_train'],
                data_splits['X_test'], data_splits['y_test'],
                stopwords_list, model_name=model_name
            )
            
            # Evaluar modelo en conjunto de prueba
            metrics = evaluate_model(model, data_splits['X_test'], data_splits['y_test'])
            
            # Registrar métricas en MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
                
            # Inferir firma del modelo
            signature = infer_signature(
                data_splits['X_train'].iloc[:5], 
                model.predict(data_splits['X_train'].iloc[:5])
            )
            
            # Registrar modelo en MLflow
            mlflow.sklearn.log_model(
                model, 
                model_name, 
                signature=signature,
                registered_model_name=model_name
            )
            
            # Guardar artefactos adicionales
            artifacts_dir = save_model_artifacts(
                model, model_name, version, metrics, 
                data_signature={"input_example": data_splits['X_train'].iloc[0]}
            )
            
            # Generar model card
            model_info = {
                "name": model_name,
                "version": version,
                "description": f"Modelo de clasificación de texto entrenado con {data_splits['X_train'].shape[0]} ejemplos",
                "training_data": {
                    "rows": data.shape[0],
                    "columns": data.shape[1],
                    "date": datetime.now().strftime("%Y-%m-%d")
                },
                "parameters": model.get_params() if hasattr(model, 'get_params') else {}
            }
            
            if artifacts_dir:
                generate_model_card(model_info, metrics, f"{artifacts_dir}/model_card.json")
        
        # 5. Comparar con modelo campeón si existe
        try:
            champion_model = get_champion_model(model_name=model_name)
            if champion_model:
                champion_metrics = evaluate_model(champion_model, data_splits['X_test'], data_splits['y_test'])
                comparison_result = compare_models(
                    champion_metrics=champion_metrics,
                    challenger_metrics=metrics,
                    primary_metric=optimization_metric,
                    threshold=champion_threshold
                )
                log_info(f"Comparación de modelos: {comparison_result}")
        except Exception as e:
            log_info(f"No se encontró modelo campeón o error en comparación: {e}")
        
        return model, metrics, run.info.run_id
    
    except Exception as e:
        log_error(f"Error en pipeline de entrenamiento: {e}")
        raise e

#get_ipython().system('jupyter nbconvert --to script NB_funciones.ipynb')

