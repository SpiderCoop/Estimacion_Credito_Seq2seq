# Librerias necesarias ---------------------------------------------------

import os
import pandas as pd
import numpy as np

import random

from datetime import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.frequencies import to_offset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

# Clase -------------------------------------------------------------------------

# Clase para estructurar el dataset de series temporales multivariadas
class MultivariateTimeSeriesDataset(Dataset):
    def __init__(self, df:pd.DataFrame, target_cols:list, exog_cols:list, hist_len:int, horizon:int,
                 num_samples:int=None, shock_dict:dict=None, mean=None, std=None, lag:int=24):
        """
        This class takes a DataFrame with time series data and prepares it for training a model. 
        Especially useful for Deep Learning models that require fixed-length input sequences and can handle multiple time series.

        Args:
            df (DataFrame): dataframe con las series de tiempo como columnas e índices de fecha tipo datetime
            target_cols (list): lista de nombres de columnas objetivo (series a predecir)
            exog_cols (list): lista de nombres de columnas exógenas (variables explicativas)
            hist_len (int): longitud de la secuencia de entrada (n pasos pasados)
            horizon (int): pasos futuros a predecir (m pasos hacia adelante)
            num_samples (int): número de muestras a generar (si None, usa todas las posibles)
            shock_dict (dict): diccionario con shocks (nombre: fecha) para agregar variables dummies. Se añade una columna por shock con 1 a partir del timestamp, 0 en caso contrario.
            mean (float): media para normalización. Ayuda para mentener parametros de normalizacion (si None, se calcula del DataFrame)
            std (float): desviación estándar para normalización. Ayuda para mentener parametros de normalizacion (si None, se calcula del DataFrame)
            lag (int): número de pasos hacia atrás para incluir como variable exógena (default 24 meses)
        
        Returns:
            x_hist (Tensor): secuencia de entrada histórica (shape: hist_len, num_features)
            x_fut (Tensor): secuencia de entrada futura (shape: horizon, num_features)
            y_hist (Tensor): variable objetivo histórica (shape: hist_len, 1)
            y_fut (Tensor): variable objetivo futura (shape: horizon, 1)
            series_idx (Tensor): índice de la serie para embeddings (shape: 1)

        """

        # Normalizamos las series
        self.mean = df.mean(axis=0) if mean is None else mean
        self.std = df.std(axis=0) if std is None else std
        self.df = (df - self.mean) / self.std

        # Incorporamos una variable con rezago para capturar la tendencia
        self.lag = lag

        # Guardamos el índice original para poder acceder a los timestamps
        self.original_index = self.df.index.copy()

        # Periodos de hisoria y horizonte de pronóstico
        self.hist_len = hist_len
        self.horizon = horizon
        self.max_idx = len(self.df) - self.hist_len - self.horizon - self.lag

        # Definimos el número de muestras
        if num_samples is not None:
            self.num_samples = num_samples
        else:
            self.num_samples = self.max_idx
        
        # Diccionario de shocks
        self.shock_dict = shock_dict

        # Columnas objetivo y exogenas
        self.target_cols = target_cols
        self.exog_cols = exog_cols

        # Mapeo de nombres de series a índices para embeddings
        self.series_id_map = {name: idx for idx, name in enumerate(self.target_cols)}

        # Genera lista completa de muestras: (serie, índice)
        self.samples_list = []
        for series_name in self.target_cols:
            for i in range(self.max_idx):
                self.samples_list.append((series_name, i))

        self.samples_list = random.sample(self.samples_list, k=self.num_samples)      


    def __len__(self):
        return len(self.samples_list)


    def __getitem__(self, idx):
        
        series_name, start_idx = self.samples_list[idx]

        # Se establece el índice de la serie para el embedding
        series_idx = self.series_id_map[series_name]

        # Obtenemos las ventanas de acuerdo al indice
        hist_slice = slice(start_idx + self.lag, start_idx + self.hist_len + self.lag)
        fut_slice = slice(start_idx + self.hist_len + self.lag, start_idx + self.hist_len + self.horizon + self.lag)

        hist_df = self.df.iloc[hist_slice]
        fut_df = self.df.iloc[fut_slice]

        # Obtenemos variables de tiempo
        hist_index = self.original_index[hist_slice]
        fut_index = self.original_index[fut_slice]
        
        time_hist = np.array([np.sin(2 * np.pi * hist_index.month / 12),
                              np.cos(2 * np.pi * hist_index.month / 12),
                              np.sin(2 * np.pi * hist_index.quarter / 4),
                              np.cos(2 * np.pi * hist_index.quarter / 4),
                              ]).T.astype(np.float32)
        
        time_fut = np.array([np.sin(2 * np.pi * fut_index.month / 12),
                             np.cos(2 * np.pi * fut_index.month / 12),
                             np.sin(2 * np.pi * fut_index.quarter / 4),
                             np.cos(2 * np.pi * fut_index.quarter / 4),
                             ]).T.astype(np.float32)
        
        
        # Obtenemos variables dummies para capturar cambios de tendencia o shocks
        if self.shock_dict is not None:
            for skock_name, date in self.shock_dict.items():
                date = pd.to_datetime(date)
                time_hist = np.concatenate((time_hist, (hist_index >= date).astype(np.float32).reshape(-1, 1)), axis=1)
                time_fut = np.concatenate((time_fut, (fut_index >= date).astype(np.float32).reshape(-1, 1)), axis=1)


        # Variables exogenas
        x_hist = hist_df[self.exog_cols].values.astype(np.float32)
        x_fut = fut_df[self.exog_cols].values.astype(np.float32)

        # Concatenamos las variables exogenas con las de tiempo
        x_hist = np.concatenate((x_hist, time_hist), axis=1)
        x_fut = np.concatenate((x_fut, time_fut), axis=1)

        
        # Variable objetivo con retraso de 24 meses para incorporar como variable exógena
        if self.lag > 0:
            hist_lag = slice(start_idx, start_idx + self.hist_len)
            fut_lag = slice(start_idx + self.hist_len, start_idx + self.hist_len + self.horizon)

            x_hist = np.concatenate((x_hist, self.df.iloc[hist_lag][series_name].values.reshape(-1, 1).astype(np.float32)), axis=1)
            x_fut = np.concatenate((x_fut, self.df.iloc[fut_lag][series_name].values.reshape(-1, 1).astype(np.float32)), axis=1)


        # Variables objetivo
        y_hist = hist_df[series_name].values.astype(np.float32)
        y_fut = fut_df[series_name].values.astype(np.float32)

        return (
            torch.tensor(x_hist).reshape(self.hist_len, -1),
            torch.tensor(x_fut).reshape(self.horizon, -1),
            torch.tensor(y_hist).reshape(self.hist_len, -1),
            torch.tensor(y_fut).reshape(self.horizon, -1),
            torch.tensor(series_idx).long()
            )    






# Clase para el modelo Seq2Seq
class Seq2seq_MultiHorizon_Quantile_Recurrent_Forecaster(pl.LightningModule):
    """
    Modelo Seq2Seq con predicción multi-horizonte y cuantílica para series de tiempo multivariadas.

    Este modelo combina un codificador LSTM, embeddings de identificación de serie, MLPs para contextos
    globales y locales, y una capa de salida que estima múltiples cuantiles simultáneamente. Está diseñado 
    para incorporar variables exógenas, estacionales, shocks estructurales y rezagos.

    Args:
        hist_len (int): Longitud de la ventana histórica utilizada como entrada.
        horizon (int): Horizonte de predicción (número de pasos futuros).
        input_size (int): Número total de variables de entrada (exógenas + opcionales).
        output_size (int): Número de variables objetivo.
        hidden_size (int): Tamaño del estado oculto del LSTM.
        num_layers (int): Número de capas en el LSTM.
        global_mlp_layers (int): Número de unidades ocultas en el MLP global.
        ct_size (int): Tamaño del contexto temporal local.
        ca_size (int): Tamaño del contexto global.
        local_mlp_layers (int): Número de unidades ocultas en el MLP local.
        num_embeddings (int): Número de series distintas (para embeddings).
        embedding_dim (int): Dimensión del embedding de serie.
        quantiles (List[float]): Lista de cuantiles a predecir (ej. [0.1, 0.5, 0.9]).
        min_hist_len (int): Longitud mínima de historia para entrenamiento.
        learning_rate (float): Tasa de aprendizaje.

    Attributes:
        series_embedding (nn.Embedding): Capa de embedding para identificación de series.
        encoder (nn.LSTM): Codificador LSTM para historia.
        global_mlp (nn.Sequential): Red MLP para contextos globales.
        local_mlp (nn.Sequential): Red MLP para contextos locales.
    """

    def __init__(self, hist_len, horizon, input_size, output_size, 
                 hidden_size = 128, num_layers = 1, 
                 global_mlp_layers = 200, ct_size=8, ca_size=8, local_mlp_layers = 200,
                 num_embeddings=1, embedding_dim=4,
                 quantiles=[0.1,0.5,0.9],
                 min_hist_len=24, 
                 learning_rate=1e-3):
        
        super().__init__()

        # Variables de longitud de historia y horizonte de pronostico
        self.hist_len = hist_len
        self.horizon = horizon

        # Variables de numero de entradas y salidas
        self.input_size = input_size
        self.output_size = output_size

        # Parametros del encoder LSTM
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Parametros de las capas MLP y contextos
        self.global_mlp_layers = global_mlp_layers
        self.ct_size = ct_size
        self.ca_size = ca_size
        self.local_mlp_layers = local_mlp_layers

        # Parametros de embedding para identificacion de series 
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim 
        
        # Parámetros de cuantiles
        self.quantiles = quantiles
        self.nq = len(quantiles)

        # Parámetro auxiliar para entrenamiento
        self.min_hist_len = min_hist_len

        # Parámetros de optimización
        self.learning_rate = learning_rate

        # Inicializacion de atributos que se deben de definir de acuerdo con el entrenamiento para manetener cohesion
        self.target_cols = None
        self.exog_cols = None
        self.mean = None
        self.std = None
        self.shock_dict = None
        self.series_id_map = None
        self.lag = 0
    
        # Embedding para identificacion de series
        self.series_embedding = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim) 

        # Encoder
        self.encoder = nn.LSTM(input_size = self.input_size + self.embedding_dim,
                               hidden_size = self.hidden_size,
                               num_layers = self.num_layers,
                               batch_first = True)
        
        # Global context
        self.global_inputs = self.hidden_size + self.horizon * (self.input_size - self.output_size + self.embedding_dim)
        self.global_outputs = self.ca_size + self.horizon * self.ct_size
        self.global_mlp = nn.Sequential(
            nn.Linear(self.global_inputs, self.global_mlp_layers),
            nn.ReLU(),
            nn.Linear(self.global_mlp_layers, self.global_outputs))
        
        # Local context
        self.local_inputs = self.input_size - self.output_size + self.ct_size + self.ca_size + self.embedding_dim
        self.local_outputs = self.output_size * self.nq
        self.local_mlp = nn.Sequential(
            nn.Linear(self.local_inputs, self.local_mlp_layers),
            nn.ReLU(),
            nn.Linear(self.local_mlp_layers, self.local_outputs))
    
    
    def configure_optimizers(self):
        """
        Configura el optimizador para el entrenamiento del modelo.

        Returns:
            torch.optim.Optimizer: Optimizador Adam con la tasa de aprendizaje especificada.
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


    def forward(self, x_hist, x_fut, y_hist, series_idx, hidden=None):
        """
        Propagación hacia adelante del modelo para generar predicciones.

        Args:
            x_hist (Tensor): Variables exógenas históricas. Shape (batch_size, hist_len, input_size - output_size).
            x_fut (Tensor): Variables exógenas futuras. Shape (batch_size, horizon, input_size - output_size).
            y_hist (Tensor): Variable objetivo histórica. Shape (batch_size, hist_len, output_size).
            series_idx (Tensor): Índices de series para embeddings. Shape (batch_size,).
            hidden (tuple, optional): Estado oculto inicial del LSTM.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: 
                - y_hat: Predicción de cuantiles. Shape (B, horizon, output_size, nq).
                - hn: Último estado oculto del encoder LSTM.
                - cn: Último estado de celda del encoder LSTM.
        """
        
        # Obtenemos las dimensiones de las entradas
        B, hist_len, _ = x_hist.shape
        _, horizon, _ = x_fut.shape
        device = x_hist.device

        # Obtenemos embeddings para las series
        emb = self.series_embedding(series_idx).reshape(B,self.embedding_dim)
        emb = emb.unsqueeze(1)                  # (B, 1, embed_dim)
        emb_hist = emb.repeat(1, hist_len, 1)   # (B, hist_len, embed_dim)
        emb_fut = emb.repeat(1, horizon, 1)     # (B, horizon, embed_dim)

        # Concatenamos las variables exogenas con los embeddings
        x_hist = torch.cat([x_hist, emb_hist], dim=2)   # (B, hist_len, input_size - output_size + embed_dim)
        x_fut = torch.cat([x_fut, emb_fut], dim=2)      # (B, horizon, input_size - output_size + embed_dim)
        x_fut_vec = x_fut.reshape(B, -1)                # Aplanamos (B, horizon * (input_size - output_size + embed_dim))

        # Concatenamos variables exogenas y objetivos para ingresar al encoder
        encoder_input = torch.cat([x_hist, y_hist], dim=2)  # (B, hist_len, input_size + emdeb_dim)
        
        # Calculamos el estado oculto inicial
        encoder_outputs, (hn, cn) = self.encoder(encoder_input, hidden)
        hn_s = hn[-1]   # Quitamos la dimesion de capas (B, hidden_size)

        # Concatenamos el estado oculto y el vector de variables exogenas futuras para calcular los contextos globales y locales
        global_input = torch.cat([hn_s, x_fut_vec], dim=1)  # (B, hidden_size + horizon * (input_size - output_size + embed_dim))
        contexts = self.global_mlp(global_input)            # (B, horizon * ct_size + ca_size)
        
        # Separamos los contextos globales(ca) y loclaes (ct)
        ca = contexts[:, :self.ca_size].unsqueeze(1).expand(-1, self.horizon, -1)   # (B, horizon, ca_size)
        ct = contexts[:, self.ca_size:].reshape(-1, self.horizon, self.ct_size)     # (B, horizon, ct_size)

        # Concatenamos los contextos globales y locales con las variables exogenas futuras para ingresar al MLP local y hacer la prediccion
        local_input = torch.cat([ct,ca,x_fut],dim=2)    # (B, horizon, input_size - output_size + ct_size + ca_size + embed_dim)
        y_hat = self.local_mlp(local_input)     # (B, horizon, output_size * nq)

        # Verificamos que las dimensiones sean las correctas
        y_hat = y_hat.view(B, horizon, self.output_size, self.nq) # (B, horizon, output_size, nq)

        return y_hat, hn, cn
    


    def quantile_loss(self, y_true, y_hat):
        """
        Calcula la pérdida de cuantiles (pinball loss) entre la predicción y el valor real.

        Args:
            y_true (Tensor): Valores verdaderos. Shape (batch_size, horizon, output_size).
            y_hat (Tensor): Predicciones de cuantiles. Shape (batch_size, horizon, output_size, nq).

        Returns:
            Tensor: Valor escalar de la pérdida promedio.
        """
        # y_true, y_hat and self.quantiles need to be broadcastable: https://pytorch.org/docs/stable/notes/broadcasting.html

        zeros = torch.zeros_like(y_hat)
        quantiles = torch.tensor(self.quantiles, device=y_true.device)
        y_true = y_true.unsqueeze(-1)

        # Calculamos el error entre la predicción y el valor verdadero
        error = y_true - y_hat
        loss = torch.mean(quantiles * torch.maximum(error, zeros) + (1 - quantiles) * torch.maximum(-error, zeros))
        
        return  loss


    def training_step(self, batch, batch_idx):
        """
        Paso de entrenamiento para una secuencia completa. Calcula la pérdida acumulada sobre múltiples ventanas de tiempo.

        Args:
            batch (Tuple[Tensor]): Tupla con x_hist, x_fut, y_hist, y_fut, series_idx.
            batch_idx (int): Índice del batch.

        Returns:
            Tensor: Pérdida media del batch.
        """

        x_hist, x_fut, y_hist, y_fut, series_idx = batch

        # Concatenamos ventanas de historia y futuro
        x_cat = torch.cat([x_hist, x_fut], dim=1) # (B, hist_len + horizon, input_size - output_size)
        y_cat = torch.cat([y_hist, y_fut], dim=1) # (B, hist_len + horizon, output_size)

        # Obtenemos las ventanas
        x_cat_mh = x_cat[:, :self.min_hist_len, :]
        x_cat_mf = x_cat[:, self.min_hist_len:self.min_hist_len+self.horizon, :]
        y_cat_mh = y_cat[:, :self.min_hist_len, :]
        y_cat_mf = y_cat[:, self.min_hist_len:self.min_hist_len+self.horizon, :] # (B, horizon, output_size)

        # Inicializamos el estado oculto
        y_hat, hn, cn = self(x_cat_mh, x_cat_mf, y_cat_mh, series_idx)
        
        # Inicio de la función de pérdida
        loss = self.quantile_loss(y_cat_mf, y_hat)

        # Iteramos sobre la secuencia
        for i in range(self.min_hist_len,self.hist_len):
            # Actualizamos los estados ocultos
            x_hist_i = x_cat[:, i:i+1, :]
            x_fut_i = x_cat[:, i+1:i+1+self.horizon, :]
            y_hist_i = y_cat[:, i:i+1, :]
            y_fut_i = y_cat[:, i+1:i+1+self.horizon, :]

            # Forward en el siguiente paso de tiempo
            y_hat, hn, cn = self(x_hist=x_hist_i, x_fut=x_fut_i, y_hist=y_hist_i, series_idx=series_idx, hidden=(hn, cn))  # (B, horizon, output_size, nq)

            # Acumula perdida en el siguiente paso del tiempo
            loss += self.quantile_loss(y_fut_i, y_hat)

        # Promediamos la perdida por el numero de pasos de tiempo
        loss = loss / (self.hist_len - self.min_hist_len + 1)
        
        # Log the training loss
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        
        return loss
    


    def validation_step(self, batch, batch_idx):
        """
        Paso de validación. Evalúa la pérdida de cuantiles en el conjunto de validación.

        Args:
            batch (Tuple[Tensor]): Tupla con x_hist, x_fut, y_hist, y_fut, series_idx.
            batch_idx (int): Índice del batch.

        Returns:
            Tensor: Pérdida de validación.
        """

        x_hist, x_fut, y_hist, y_fut, series_idx = batch

        y_hat, hn, cn = self(x_hist, x_fut, y_hist, series_idx)
        
        loss = self.quantile_loss(y_fut, y_hat)

        self.log('val_QL', loss, prog_bar=True)

        return loss



    def predict(self, df: pd.DataFrame, target_col: str, initial_pred_date: pd.Timestamp | str):
        """
        Genera predicciones futuras para una serie temporal a partir de un DataFrame.

        La función normaliza los datos, genera variables temporales (mes, trimestre, shocks),
        aplica embeddings de serie, y realiza la inferencia del modelo. Devuelve un DataFrame
        con las predicciones para los cuantiles especificados.

        Es necesario definir los atributos de media, desviacion estandar, diccionario de shocks,
        el mapeo de las series para embedding y el lag utilizado durante el entrenamiento para 
        dar una prediccion coherente

        Args:
            df (pd.DataFrame): DataFrame con las variables normalizadas y un índice temporal.
            target_col (str): Nombre de la variable objetivo a predecir.
            initial_pred_date (Timestamp | str): Fecha a partir de la cual se inicia la predicción.

        Returns:
            pd.DataFrame: DataFrame con la predicción para cada cuantil y cada fecha futura.
        """

        self.eval()
        with torch.no_grad():
            
            # Validamos que este en orden
            df = df.sort_index()

            # Convertimos la fecha si no lo es ya
            if not isinstance(initial_pred_date, pd.Timestamp):
                if isinstance(initial_pred_date, str):
                    initial_pred_date = pd.to_datetime(initial_pred_date)
                elif isinstance(initial_pred_date, (int, float)):
                    initial_pred_date = pd.to_datetime(initial_pred_date, unit='s')
                else:
                    raise ValueError("initial_pred_date debe ser un string, int, float o pd.Timestamp.")
                

            # Inferimos la frecuencia del índice
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("El índice del DataFrame debe ser un pd.DatetimeIndex.")
            if not df.index.freq:
                raise ValueError("El índice del DataFrame debe tener una frecuencia definida.")
            
            # Obtener la frecuencia inferida del índice
            freq = df.index.inferred_freq
            offset = to_offset(freq)

            # Validamos que la fecha de inicio de predicción esté dentro del indice del DataFrame para evitar errores por la frecuencia
            initial_pred_date_idx = df.index.get_indexer([initial_pred_date], method='nearest')
            initial_pred_date_idx = initial_pred_date_idx[0] if initial_pred_date_idx.size > 0 else None
            initial_pred_date = df.index[initial_pred_date_idx] if initial_pred_date_idx is not None else None
            last_obs_date = initial_pred_date - offset

            # Reindexamos el dataframe para asegurar que esten todas los renglones de las fechas a predecir aunque generen nan
            df = df.reindex(pd.date_range(start=df.index.min(), end=(initial_pred_date + offset*self.horizon), freq=offset))

            # Definimos el slice de historia y futuro
            hist_slice = slice(initial_pred_date_idx - self.hist_len, initial_pred_date_idx)
            fut_slice = slice(initial_pred_date_idx, initial_pred_date_idx + self.horizon)

            # Validamos que las columnas de df coincidan con las medias y desviaciones estandar
            if not df.columns.equals(self.mean.index):
                df = df.reindex(columns=self.mean.index)
            
            # Noralizamos las variables
            df_norm = (df - self.mean) / self.std

            # Extraer ventanas de datos
            x_hist_np = df_norm.iloc[hist_slice][self.exog_cols].values.astype(np.float32)
            x_fut_np = df_norm.iloc[fut_slice][self.exog_cols].values.astype(np.float32)
            y_hist_np = df_norm.iloc[hist_slice][target_col].values.astype(np.float32)

            # Si la lista de columnas exogenas no esta vacia, validamos que las dimensiones sean correctas
            if len(self.exog_cols) > 0:
                if x_hist_np.shape[1] != len(self.exog_cols) or x_fut_np.shape[1] != len(self.exog_cols):
                    raise ValueError("Las dimensiones de las columnas exogenas no coinciden con las esperadas.")

            if x_hist_np.shape[0] < self.hist_len or x_fut_np.shape[0] < self.horizon:
                raise ValueError("Datos insuficientes para la predicción.")
            
            # Variables de tiempo
            time_idx_hist = df_norm.iloc[hist_slice].index
            time_idx_fut = df_norm.iloc[fut_slice].index

            # Obtenemos las variables de tiempo
            time_hist = np.array([np.sin(2 * np.pi * time_idx_hist.month / 12),
                                  np.cos(2 * np.pi * time_idx_hist.month / 12),
                                  np.sin(2 * np.pi * time_idx_hist.quarter / 4),
                                  np.cos(2 * np.pi * time_idx_hist.quarter / 4),
                                  ]).T.astype(np.float32)
        
            time_fut = np.array([np.sin(2 * np.pi * time_idx_fut.month / 12),
                                 np.cos(2 * np.pi * time_idx_fut.month / 12),
                                 np.sin(2 * np.pi * time_idx_fut.quarter / 4),
                                 np.cos(2 * np.pi * time_idx_fut.quarter / 4),
                                 ]).T.astype(np.float32)
            

            # Obtenemos variables dummies para capturar cambios de tendencia o shocks
            if self.shock_dict is not None:
                for shock_name, date in self.shock_dict.items():
                    date = pd.to_datetime(date)
                    time_hist = np.concatenate((time_hist, (time_idx_hist >= date).astype(np.float32).reshape(-1,1)), axis=1)
                    time_fut = np.concatenate((time_fut, (time_idx_fut >= date).astype(np.float32).reshape(-1,1)), axis=1)


            # Tensores
            x_hist = torch.tensor(x_hist_np, dtype=torch.float32).unsqueeze(0) # (1, hist_len, input_size - output_size)
            x_fut = torch.tensor(x_fut_np, dtype=torch.float32).unsqueeze(0)
            y_hist = torch.tensor(y_hist_np, dtype=torch.float32).unsqueeze(0)  # (1, hist_len, output_size)
            time_hist = torch.tensor(time_hist, dtype=torch.float32).unsqueeze(0)
            time_fut = torch.tensor(time_fut, dtype=torch.float32).unsqueeze(0)

            # Aseguramos que las dimensiones sean correctas
            x_hist = x_hist.view(1, self.hist_len, -1)  # (1, hist_len, input_size - output_size)
            x_fut = x_fut.view(1, self.horizon, -1)
            y_hist = y_hist.view(1, self.hist_len, -1)
            time_hist = time_hist.view(1, self.hist_len, -1)  # (1, hist_len, num_time_features)
            time_fut = time_fut.view(1, self.horizon, -1)

            # Validamos que todos los tensores se hayan construido bien
            assert not torch.isnan(x_hist).any(), "El tensor de historia de las variables exogenas contiene NaNs"
            assert not torch.isnan(x_fut).any(), "El tensor de futuro de las variables exogenas contiene NaNs"
            assert not torch.isnan(y_hist).any(), "El tensor de historia de las variables objetivo contiene NaNs"
            assert not torch.isnan(time_hist).any(), "El tensor de historia de las variables de tiempo contiene NaNs"
            assert not torch.isnan(time_fut).any(), "El tensor de futuro de las variables tiempo contiene NaNs"

            # Concatenamos las variables de tiempo
            x_hist = torch.cat((x_hist, time_hist), dim=2) # (1, hist_len, input_size - output_size + num_time_features)
            x_fut = torch.cat((x_fut, time_fut), dim=2)

            # Variable objetivo con retraso de 24 meses para incorporar como variable exógena
            if self.lag > 0:
                hist_lag = slice(initial_pred_date_idx - self.hist_len - self.lag, initial_pred_date_idx - self.lag)
                fut_lag = slice(initial_pred_date_idx - self.lag, initial_pred_date_idx + self.horizon - self.lag)

                # Verificamos la longitud 
                assert (hist_lag.start >= 0), f"Datos insuficientes para la predicción. Es necesario agregar {np.abs(hist_lag.start)} periodos adicionales hacia atras"

                x_hist_lag = df_norm.iloc[hist_lag][target_col].values.astype(np.float32)
                x_fut_lag = df_norm.iloc[fut_lag][target_col]

                if self.lag < self.horizon:
                    x_fut_lag = x_fut_lag.interpolate(method='linear')
                
                x_fut_lag = x_fut_lag.values.astype(np.float32)

                x_hist_lag = torch.tensor(x_hist_lag, dtype=torch.float32).unsqueeze(0)
                x_fut_lag = torch.tensor(x_fut_lag, dtype=torch.float32).unsqueeze(0)

                x_hist_lag = x_hist_lag.view(1, self.hist_len, -1)
                x_fut_lag = x_fut_lag.view(1, self.horizon, -1)

                # Validamos que los tensores esten bien construidos
                assert not torch.isnan(x_hist_lag).any(), "El tensor de historia de las variables de rezago contiene NaNs"
                assert not torch.isnan(x_fut_lag).any(), "El tensor de futuro de las variables de rezago contiene NaNs"

                # Concatenamos las variables de lag
                x_hist = torch.cat((x_hist, x_hist_lag), dim=2) # (1, hist_len, input_size - output_size + num_time_features + 1)
                x_fut = torch.cat((x_fut, x_fut_lag), dim=2) # (1, horizon, input_size - output_size + num_time_features + 1)


            # Convertimos el nombre de la serie objetivo a un índice para el embedding
            series_idx = torch.tensor([self.series_id_map[target_col]], dtype=torch.long)  # (1,)

            # Predicción
            y_hat, _, _ = self(x_hist, x_fut, y_hist, series_idx)  # (B, horizon, output_size, nq)
            y_hat = y_hat[0].cpu()  # (horizon, output_size, nq)

            # Expandimos las medias y desviaciones estandar para la predicción
            mean_y = torch.tensor(self.mean[target_col].astype(np.float32))
            std_y = torch.tensor(self.std[target_col].astype(np.float32))
            mean_expanded = mean_y.view(1, self.output_size, 1).repeat(self.horizon, 1, self.nq)
            std_expanded = std_y.view(1, self.output_size, 1).repeat(self.horizon, 1, self.nq)

            # Invertimos la normalización
            y_hat_inv = y_hat * std_expanded + mean_expanded

            # Construcción del DataFrame de salida
            fechas = pd.date_range(start=initial_pred_date, periods=self.horizon, freq=offset)
            df_pred = pd.DataFrame(y_hat_inv.reshape(self.horizon, -1).numpy(),
                            columns=[f'{target_col}_{int(q*100)}' for q in self.quantiles],
                            index=fechas)
            
            # Agregamos la última observacion para graficar mejor
            df_pred.loc[last_obs_date, :] = df.loc[last_obs_date, target_col]
            df_pred = df_pred.sort_index()
            
            return df_pred
        
    def save_model(self, path:str):
        """
        Guarda un modelo con sus atributos.

        Args:
            path (str): Ruta del archivo .pth a crear.

        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'hist_len': self.hist_len,
            'horizon': self.horizon,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'global_mlp_layers': self.global_mlp_layers,
            'ct_size': self.ct_size,
            'ca_size': self.ca_size,
            'local_mlp_layers': self.local_mlp_layers,
            'num_embeddings': self.num_embeddings,
            'embedding_dim': self.embedding_dim,
            'min_hist_len': self.min_hist_len,
            'quantiles': self.quantiles,
            'learning_rate': self.learning_rate,
            'target_cols': self.target_cols,
            'exog_cols': self.exog_cols,
            'mean': self.mean,
            'std': self.std,
            'shock_dict': self.shock_dict,
            'series_id_map': self.series_id_map,
            'lag': self.lag,
        }, path)

        print(f"Modelo y atributos guardados en {path}")

    @classmethod
    def load_model(cls, path:str):
        """
        Carga un modelo previamente guardado junto con sus atributos.

        Args:
            path (str): Ruta al archivo .pth guardado.

        Returns:
            Seq2seq_MultiHorizon_Quantile_Recurrent_Forecaster: Instancia del modelo cargado.
        """
        checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=False)

        # Crear una nueva instancia del modelo con los hiperparámetros guardados
        model = cls(
            hist_len=checkpoint['hist_len'],
            horizon=checkpoint['horizon'],
            input_size=checkpoint['input_size'],
            output_size=checkpoint['output_size'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers'],
            global_mlp_layers=checkpoint['global_mlp_layers'],
            ct_size=checkpoint['ct_size'],
            ca_size=checkpoint['ca_size'],
            local_mlp_layers=checkpoint['local_mlp_layers'],
            num_embeddings=checkpoint['num_embeddings'],
            embedding_dim=checkpoint['embedding_dim'],
            quantiles=checkpoint['quantiles'],
            min_hist_len=checkpoint['min_hist_len'],
            learning_rate=checkpoint['learning_rate']
        )

        # Cargar los pesos
        model.load_state_dict(checkpoint['model_state_dict'])

        # Restaurar atributos adicionales
        model.target_cols = checkpoint.get('target_cols')
        model.exog_cols = checkpoint.get('exog_cols')
        model.mean = checkpoint.get('mean')
        model.std = checkpoint.get('std')
        model.shock_dict = checkpoint.get('shock_dict')
        model.series_id_map = checkpoint.get('series_id_map')
        model.lag = checkpoint.get('lag', 0)

        print(f"Modelo cargado desde {path}")
        return model

