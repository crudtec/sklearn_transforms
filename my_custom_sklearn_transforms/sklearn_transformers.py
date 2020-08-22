from sklearn.base import BaseEstimator, TransformerMixin
import math
import pandas as pd
import imblearn
from imblearn.over_sampling import SMOTE

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

   
class PreencherDados(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df_data_2 = X.copy()
        # Setar Lingua Inglesa Nula como 0 - Considero que ele nao fala outra lingua
        df_data_3 = df_data_2
        df_data_3.update(df_data_3['INGLES'].fillna(0))
        # Setar Reprovações nulas como 0 - Considero que nunca foi reprovado
        df_data_3.update(df_data_3['REPROVACOES_DE'].fillna(0))
        df_data_3.update(df_data_3['REPROVACOES_EM'].fillna(0))
        df_data_3.update(df_data_3['REPROVACOES_MF'].fillna(0))
        df_data_3.update(df_data_3['REPROVACOES_GO'].fillna(0))
        # Setar NOTA_GO de acordo com media geral para as notas zeradas
        Filtro  = df_data_2['NOTA_GO'] > 0
        DS = df_data_2[Filtro]
        Media = DS['NOTA_GO'].mean()
        df_data_3.update(df_data_3['NOTA_GO'].fillna(Media))
        # Setar NOTA_DE de acordo com media geral  para as notas zeradas
        Filtro  = df_data_2['NOTA_DE'] > 0
        DS = df_data_2[Filtro]
        Media = DS['NOTA_DE'].mean()
        df_data_3.update(df_data_3['NOTA_DE'].fillna(Media))
        # Setar NOTA_EM de acordo com media geral  para as notas zeradas
        Filtro  = df_data_2['NOTA_EM'] > 0
        DS = df_data_2[Filtro]
        Media = DS['NOTA_EM'].mean()
        df_data_3.update(df_data_3['NOTA_EM'].fillna(Media))
        # Setar NOTA_MF de acordo com media geral  para as notas zeradas
        Filtro  = df_data_2['NOTA_MF'] > 0
        DS = df_data_2[Filtro]
        Media = DS['NOTA_MF'].mean()
        df_data_3.update(df_data_3['NOTA_MF'].fillna(Media))
        # Setar H_AULA_PRES de acordo com media 
        Filtro  = df_data_2['H_AULA_PRES'] > 0
        DS = df_data_2[Filtro]
        Media = DS['H_AULA_PRES'].mean()
        df_data_3.update(df_data_3['H_AULA_PRES'].fillna(Media))
        # Setar TAREFAS_ONLINE de acordo com media geral
        Filtro  = df_data_2['TAREFAS_ONLINE'] > 0
        DS = df_data_2[Filtro]
        Media = DS['TAREFAS_ONLINE'].mean()
        df_data_3.update(df_data_3['TAREFAS_ONLINE'].fillna(Media))
        # Setar FALTAS de acordo com media geral
        Filtro  = df_data_2['FALTAS'] > 0
        DS = df_data_2[Filtro]
        Media = DS['FALTAS'].mean()
        df_data_3.update(df_data_3['FALTAS'].fillna(Media))       
        return df_data_3 
    
class MediasRelacionadas(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        #adiciona coluna de medias relacionadas
        df = X.copy()
        df['REPROVACOES_H'] = df['REPROVACOES_DE'] + df['REPROVACOES_EM']
        df['REPROVACOES_E'] = df['REPROVACOES_MF'] + df['REPROVACOES_GO']
        df['MEDIA_H'] = (df['NOTA_DE']+df['NOTA_EM'])/2
        df['MEDIA_E'] = (df['NOTA_MF']+df['NOTA_GO'])/2

        return df

class SmoteBalancear(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        X_resampled, y_resampled = SMOTE().fit_resample(X, y)
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        return X_resampled, y_resampled
    
    def transform(self, X):
        #adiciona coluna de medias relacionadas
        df = X.copy()
        return df
