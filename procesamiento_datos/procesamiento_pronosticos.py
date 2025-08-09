# Librerias necesarias ---------------------------------------------------

import numpy as np
import pandas as pd

# Configuracion inicial -------------------------------------------------------------------------


def descarga_pron_trimestral(banxico_api, serie:str, estadistico:str = 'mediana'):
    # Tomamos el trimestre actual para definir el trimestre de pronostico
    fecha_actual = pd.Timestamp.now()
    trim_actual = fecha_actual.quarter

    if trim_actual == 1:
        fecha_inicio = pd.Timestamp(fecha_actual.year, 3, 1)
    elif trim_actual == 2:
        fecha_inicio = pd.Timestamp(fecha_actual.year, 6, 1)
    elif trim_actual == 3:
        fecha_inicio = pd.Timestamp(fecha_actual.year, 9, 1)

    if serie == 'PIB':
        # Lectura de las series de pronosticos del PIB trimestrales
        delay = pd.DateOffset(months=-3)  # Retraso de 3 meses para el inicio del pronostico
        pron_media_trim_id = {'SR14475':'pib t-1',
                              'SR14482':'pib t',
                              'SR14489':'pib t+1',
                              'SR14496':'pib t+2',
                              'SR14503':'pib t+3',
                              'SR14510':'pib t+4',
                              'SR14517':'pib t+5',
                              'SR14524':'pib t+6',
                              'SR14531':'pib t+7',
                              'SR15906':'pib t+8'}
        
        pron_mediana_trim_id = {'SR14476':'pib t-1',
                                'SR14483':'pib t',
                                'SR14490':'pib t+1',
                                'SR14497':'pib t+2',
                                'SR14504':'pib t+3',
                                'SR14511':'pib t+4',
                                'SR14518':'pib t+5',
                                'SR14525':'pib t+6',
                                'SR14532':'pib t+7',
                                'SR15907':'pib t+8'}
        
        
    elif serie == 'TIIE_Fondeo':
        # Lectura de las series de pronosticos de tasa de fondeo
        delay = pd.DateOffset(months=0)
        pron_media_trim_id = {'SR14658':'tiie t',
                              'SR14665':'tiie t+1',
                              'SR14672':'tiie t+2',
                              'SR14679':'tiie t+3',
                              'SR14686':'tiie t+4',
                              'SR14693':'tiie t+5',
                              'SR14700':'tiie t+6',
                              'SR14707':'tiie t+7',
                              'SR14714':'tiie t+8'}
        
        pron_mediana_trim_id = {'SR14659':'tiie t',
                                'SR14666':'tiie t+1',
                                'SR14673':'tiie t+2',
                                'SR14680':'tiie t+3',
                                'SR14687':'tiie t+4',
                                'SR14694':'tiie t+5',
                                'SR14701':'tiie t+6',
                                'SR14708':'tiie t+7',
                                'SR14715':'tiie t+8'}
        

    if estadistico == 'media':
        series_dict = pron_media_trim_id
    elif estadistico == 'mediana':
        series_dict = pron_mediana_trim_id

    # Series de pronosticos trimestrales del PIB
    pron_trim = banxico_api.get_data(list(series_dict.keys()), ult_obs=True)

    # Se ordenan las columnas de acuerdo a su nombre. Coincide con el orden de los trimestres 
    pron_trim = pron_trim.sort_index(axis=1)

    # Se transpone y renombran las columnas para que coincidan con los trimestres y poder concatenarlas mas facilmente
    pron_trim = pron_trim.transpose()
    pron_trim.index = pd.date_range(start=fecha_inicio + delay, periods=len(pron_trim), freq='QS-MAR')

    # Se renombra la serie como pron_trim_pib
    pron_trim.columns = [f'pron_trim_{serie.lower()}']

    # Convertir la serie a un tipo de datos numérico antes de interpolar
    pron_trim[f'pron_trim_{serie.lower()}'] = pd.to_numeric(pron_trim[f'pron_trim_{serie.lower()}'], errors='coerce')  # Convertir a numérico

    # Realizar la interpolación lineal
    pron_trim.iloc[:,0] = pron_trim.iloc[:,0].interpolate(method='linear', limit_direction='backward')

    return pron_trim


def descarga_pron_anual(banxico_api, serie:str, estadistico:str = 'mediana'):
    # Tomamos el trimestre actual para definir el trimestre de pronostico
    fecha_actual = pd.Timestamp.now()
    trim_actual = fecha_actual.quarter

    if trim_actual == 1:
        fecha_inicio = pd.Timestamp(fecha_actual.year, 3, 1)
    elif trim_actual == 2:
        fecha_inicio = pd.Timestamp(fecha_actual.year, 6, 1)
    elif trim_actual == 3:
        fecha_inicio = pd.Timestamp(fecha_actual.year, 9, 1)

    if serie == 'PIB':
        # Lectura de las series de pronosticos del PIB anuales
        delay = pd.DateOffset(months=-12)  # Retraso de 12 meses para el inicio del pronostico
        pron_media_anual_id = {'SR14440':'pib t-1',
                               'SR14447':'pib t',
                               'SR14454':'pib t+1',
                               'SR14461':'pib t+2'}
        
        pron_mediana_anual_id = {'SR14441':'pib t-1',
                                 'SR14448':'pib t',
                                 'SR14455':'pib t+1',
                                 'SR14462':'pib t+2'}
        
    elif serie == 'inflacion':
        # Lectura de las series de pronosticos de inflación
        delay = pd.DateOffset(months=0)
        pron_media_anual_id = {'SR14138':'pib t',
                               'SR14145':'pib t+1',
                               'SR14152':'pib t+2'}
        
        pron_mediana_anual_id = {'SR14139':'pib t',
                                 'SR14146':'pib t+1',
                                 'SR14153':'pib t+2'}

    if estadistico == 'media':
        series_dict = pron_media_anual_id
    elif estadistico == 'mediana':
        series_dict = pron_mediana_anual_id

    # Series de pronosticos anuales del PIB
    pron_anual = banxico_api.get_data(list(series_dict.keys()), ult_obs=True)

    # Se ordenan las columnas de acuerdo a su nombre. Coincide con el orden de los años 
    pron_anual = pron_anual.sort_index(axis=1)

    # Se transpone y renombran las columnas para que coincidan con los años y poder concatenarlas mas facilmente
    pron_anual = pron_anual.transpose()
    pron_anual.index = pd.date_range(start=fecha_inicio + delay, periods=len(pron_anual), freq='YE')

    # Se renombra la serie como pron_anual
    pron_anual.columns = [f'pron_anual_{serie.lower()}']

    # Cambiamos el indice para que sean los primeros dias del mes de la fecha de pronostico y facilitar la union mas adelante
    pron_anual.index = pd.to_datetime(pron_anual.index.to_period('M').to_timestamp())

    return pron_anual


def procesamiento_pron_pib(banxico_api, pronosticos_def:dict = None, estadistico:str = 'mediana'):
    # Descrarga de la serie y pronosticos del PIB trimestral y anual
    fecha_actual = pd.Timestamp.now()
    fecha_t_2 = fecha_actual - pd.DateOffset(years=3)
    series_dict = {'SR17622':'PIB'}

    pib = banxico_api.get_data(list(series_dict.keys()), fecha_inicio=fecha_t_2.strftime('%Y-%m-%d'))
    pib = pib.sort_index(axis=1)
    pib = pib.apply(pd.to_numeric, errors='coerce')
    pib = pib.rename(columns=series_dict)


    pron_trim_pib = descarga_pron_trimestral(banxico_api, 'PIB', estadistico = estadistico)
    pron_anual_pib = descarga_pron_anual(banxico_api, 'PIB', estadistico = estadistico)

    # Calculamos la variación anual del PIB
    pib['var_pib'] = (pib['PIB']/pib['PIB'].shift(4)-1)*100

    # Se toman los trimestres faltantes para el fin de año
    ult_trim = pd.to_datetime(pib.dropna().index).max()
    trim_t = ult_trim + pd.DateOffset(months=3)
    trim_index_t = [trim_t]

    # Itera para generar la lista de los faltantes
    while trim_t.month < 12:
        trim_t = trim_t + pd.DateOffset(months=3)
        trim_index_t.append(trim_t)

    trim_index_t = pd.to_datetime(trim_index_t)
    trim_index_t.freq = 'QS-MAR'

    # Realizamos un ajuste a los pronosticos trimestrales para que coincidan con el pronostico anual del año t

    # Obtenemos las observaciones reales
    filtro_trim_t = pd.date_range(start=trim_index_t[-1] + pd.DateOffset(months=-11), end=ult_trim, freq='QS-MAR')
    var_real_pib_t = pib.loc[filtro_trim_t,'var_pib']

    # Obtenemos los pronosticos de los trimestres que hacen falta
    pronosticos = pron_trim_pib.loc[trim_index_t].dropna()

    if pronosticos_def is not None:
        # Si se pasan pronosticos definidos, los usamos en lugar de los descargados
        suma_objetivo = 4*pronosticos_def.get('t')
    else:  
        suma_objetivo = 4*pron_anual_pib.loc[trim_index_t[-1]].iloc[0]

    # Realizamos el ajuste con respecto al promedio
    suma_obs = var_real_pib_t.sum()
    suma_estimaciones_requerida = suma_objetivo - suma_obs
    factor = suma_estimaciones_requerida / pronosticos.sum()

    pron_t_ajustados = pronosticos * factor

    # Realizamos un ajuste a los pronosticos trimestrales para que coincidan con el pronostico anual del año t+1

    # Rellanamos los datos faltantes con los pronosticos trimestrales
    trim_index_t_1 = pd.date_range(start=trim_index_t[-1] + pd.DateOffset(months=1), end=trim_index_t[-1] + pd.DateOffset(months=12), freq='QS-MAR')

    # Obtenemos los pronosticos de los trimestres que hacen falta
    pronosticos = pron_trim_pib.loc[trim_index_t_1].dropna()

    if pronosticos_def is not None:
        # Si se pasan pronosticos definidos, los usamos en lugar de los descargados
        suma_objetivo = 4*pronosticos_def.get('t_1')
    else:  
        suma_objetivo = 4* pron_anual_pib.loc[trim_index_t_1[-1]].iloc[0]

    # Realizamos el ajuste con respecto al promedio
    suma_obs = 0
    suma_estimaciones_requerida = suma_objetivo - suma_obs
    factor = suma_estimaciones_requerida / pronosticos.sum()

    pron_t_1_ajustados = pronosticos * factor

    # Concatenamos los indices
    trim_index = pd.to_datetime(list(trim_index_t) + list(trim_index_t_1))
    trim_index = pd.DatetimeIndex(trim_index).drop_duplicates().sort_values()
    trim_index.freq = 'QS-MAR'

    # Concatenamos los pronosticos trimestrales 
    var_real_pib_ajustado = pd.concat([pron_t_ajustados, pron_t_1_ajustados])

    return trim_index, var_real_pib_ajustado



