import os
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.tsa.statespace.tools as tools
from statsmodels.api import tsa
from numpy.typing import ArrayLike
from pandas import Series, DataFrame
from typing import Dict, Tuple, Union
from pmdarima.arima.utils import nsdiffs
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, pacf, acf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


warnings.filterwarnings('ignore')
args = {'fft': False, 'bartlett_confint': True, 'adjusted': False, 'missing': 'none'}


def adf_values(series, show=False):
    adf_results = adfuller(series.dropna(), autolag='AIC')
    result = {
        "Test Statistic": adf_results[0],
        "p-value": adf_results[1],
        "Lags": adf_results[2],
        "N° Obs": adf_results[3],
        "Critical Value (1%)": adf_results[4]["1%"],
        "Critical Value (5%)": adf_results[4]["5%"],
        "Critical Value (10%)": adf_results[4]["10%"]
    }
    if show:
        for key, val in result.items():
            string_size = len(str(key)) + len(str(round(val,3)))
            if string_size < 50:
                space = " "*(50 - string_size)
            print(f"{key}{space}{round(val,3)}")
    return result


def calculate_performance_metrics(df: DataFrame) -> Tuple[float, float, float, float]:
    """
    Calcula e retorna as métricas de desempenho para um modelo de previsão de séries temporais.

    Esta função calcula quatro métricas comuns:
    - MAPE (Erro Percentual Absoluto Médio): Média dos erros percentuais absolutos.
    - R² (Coeficiente de Determinação): Proporção da variância explicada pelo modelo.
    - RMSE (Raiz do Erro Quadrático Médio): Raiz quadrada da média dos quadrados dos erros.
    - MAE (Erro Médio Absoluto): Média das diferenças absolutas entre previsões e valores reais.

    Parâmetros:
    df (DataFrame): Um DataFrame do pandas contendo duas colunas - 'real' para os valores reais
                    e 'forecast' para os valores previstos pelo modelo.

    Retorna:
    Tuple[float, float, float, float]: Uma tupla contendo os valores das métricas MAPE, R², RMSE e MAE,
                                       nesta ordem.
    """
    median_ape = np.median(np.abs((df.real - df.forecast) / df.real)) * 100
    mape = np.mean(np.abs((df.real - df.forecast) / df.real)) * 100
    r2 = r2_score(df.real, df.forecast)
    rmse = mean_squared_error(df.real, df.forecast, squared=False)
    mae = mean_absolute_error(df.real, df.forecast)
    return mape, r2, rmse, mae, median_ape


def get_seasonality(series: pd.Series):
    """
    series: Série temporal já diferenciada sazonalmente e não sazonal
    """
    lim = (series.shape[0]*0.9) / 3.5
    possible_seasonalities = [i if nsdiffs(series, m=i, max_D=100) == 1 else 0 for i in range(2, int(lim))]
    s = list(
        filter(
            lambda x: x!=0, possible_seasonalities
            )
        )
    return 0 if not s else min(s)


def make_stationary(
        series: Union[Series, DataFrame], 
        season_diff: bool=True, 
        D: int=0, 
        d: int=0, 
        s: int=0
    ) -> Tuple[int, int, int]:

    df = series.copy()
    adf_test = adf_values(df)
    while adf_test["p-value"] > 0.05:
        d+=1
        df = df.diff().dropna()
        adf_test = adf_values(df)
    if season_diff:
        s, D = seasonality_diff(df)
        return make_stationary(df, False, D, d, s)
    return d, D, s


def prepare_metrics(metrics: Dict[str, any]):
    metrics["tests_data"]["date"] = metrics["tests_data"]["date"].dt.strftime("%Y-%m-%d")
    metrics["tests_data"] = metrics["tests_data"].to_dict()
    return metrics


def residual_analysis(model: SARIMAX) -> Tuple[bool, str]:
    resid = model.fit().resid
    if acorr_ljungbox(resid, lags=resid.shape[0]-1, return_df=True).dropna().lb_pvalue.min() < 0.05:
        text = "O teste estatistico indica que os residuos nao sao independentes, logo, há padroes nos residuos que o modelo nao capturou"
        return True, text
    text = "O teste estatistico indica que os residuos sao independentes, logo, nao há padroes restantes nos residuos"
    return False, text


def seasonality_diff(series: Series) -> Tuple[int, int]:
    s = get_seasonality(series)
    D = 0
    if s != 0:
        D = nsdiffs(series, m=s)
    return s, D
        

def set_history_df(f: str, setor: str, estado: Union[str,None]=None):
    if estado:
        df = pd.read_csv(f".\\bases\\{f}", sep=';').query(f"estado == '{estado}' and setor == '{setor}'").reset_index(drop=True)
        df = df[df.columns[1:]].T.iloc[1:]
    else:
        df = pd.read_csv(f".\\bases\\{f}", sep=';').query(f"setor == '{setor}'").groupby("setor").sum(numeric_only=True)
        df = df.T
    df.columns = ["setor"]
    return df


class SARIMA:
    def __init__(self, series: ArrayLike):
        self._init_variables()
        self.original_series = series
        self._setup_values()
    
    def _init_variables(self):
        self._init_sarima_orders(), self._init_sarima_series()
        self._model = None
        self._order = None
        self._seasonal_order = None
      
    def _init_sarima_orders(self):
        self._p, self._d, self._q = None, None, None # Non seasonal order
        self._P, self._D, self._Q = None, None, None # Seasonal order
        self._lags, self._s = None, None # Other important variables to SARIMA

    def _init_sarima_series(self):
        self._original_series, self._series_diff = None, None

    def _setup_values(self):
        self._setup_dDs()
        self._setup_series_diff_and_lags()
        self._setup_qQ(), self._setup_pP()
        self._setup_orders()
    
    def _setup_orders(self):
        self.order = (self.p, self.d, self.q)
        self.seasonal_order = (self.P, self.D, self.Q, self.s)

    @property
    def seasonal_order(self):
        return self._seasonal_order

    @property
    def order(self):
        return self._order
    
    @property
    def original_series(self):
        return self._original_series
    
    @property
    def lags(self):
        return self._lags

    @property
    def p(self):
        return self._p

    @property
    def d(self):
        return self._d

    @property
    def q(self):
        return self._q

    @property
    def P(self):
        return self._P

    @property
    def D(self):
        return self._D

    @property
    def Q(self):
        return self._Q

    @property
    def s(self):
        return self._s
    
    @property
    def series_diff(self):
        return self._series_diff
    
    @property
    def model(self):
        return self._model
    
    @original_series.setter
    def original_series(self, value):
        self._original_series = value

    @lags.setter
    def lags(self, value):
        self._lags = value

    @p.setter
    def p(self, value):
        self._p = value

    @d.setter
    def d(self, value):
        self._d = value

    @q.setter
    def q(self, value):
        self._q = value

    @P.setter
    def P(self, value):
        if value <= 0:
            self._P = 0
        else:
            self._P = value

    @D.setter
    def D(self, value):
        if value <= 0:
            self._D = 0
        else:
            self._D = value

    @Q.setter
    def Q(self, value):
        if value <= 0:
            self._Q = 0
        else:
            self._Q = value

    @s.setter
    def s(self, value):
        if value > 3:
            self._s = min(12, value)
        else:
            self._s = 0

    @series_diff.setter
    def series_diff(self, value):
        self._series_diff = value

    @model.setter
    def model(self, value):
        self._model = value

    @seasonal_order.setter
    def seasonal_order(self, value: Tuple[int, int, int]):
        self._seasonal_order = value

    @order.setter
    def order(self, value: Tuple[int, int, int]):
        self._order = value

    def __str__(self):
        return f"SARIMA({self._p}, {self._d}, {self._q})({self._P}, {self._D}, {self._Q}, {self._s})"
    
    def _setup_qQ(self):
        acf_results = acf(self.series_diff, alpha=0.05, nlags=self.lags, **args)[:2]
        acf_infos = {"values": acf_results[0], "confint": acf_results[1]}
        sup = (acf_infos["confint"][:, 1] - acf_infos["values"]) # SUP_acf ACF
        self.q = next((lag for lag in range(1, self.lags) if abs(acf_infos["values"][lag]) < sup[lag]), 0)
        if self.s:
            self.Q = (next((lag for lag in range(self.s, self.lags, self.s) if abs(acf_infos["values"][lag]) < sup[lag]), 0) // self.s )
        else:
            self.Q = 0

    def _setup_pP(self):
        pacf_results = pacf(self.series_diff, alpha=0.05, nlags=self.lags, method="ywm")
        pacf_infos = {"values": pacf_results[0], "confint": pacf_results[1]}
        significance_level = 1.96 / np.sqrt(len(pacf_infos["values"]))
        sup = [significance_level for i in range(len(pacf_infos["values"]))]
        self.p = next((lag for lag in range(1, self.lags) if abs(pacf_infos["values"][lag]) < significance_level), 0)
        if self.s:
            self.P = (next((lag for lag in range(self.s, self.lags, self.s) if abs(pacf_infos["values"][lag]) < sup[lag]), 0) // self.s )
        else:
            self.P = 0

    def _setup_dDs(self):
        d, D, s = make_stationary(self.original_series)
        self.d = d
        self.s = s
        self.D = D
        
    def _setup_series_diff_and_lags(self):
        self.series_diff = tools.diff(self.original_series, k_diff=self.d, k_seasonal_diff=self.D, seasonal_periods=self.s)
        self.lags = int(self.series_diff.shape[0]/2) - 1

    def evaluate_model(self):
        loops = self.test_model()
        evaluate_df = DataFrame(loops).T[["forecast", "real"]]
        evaluate_df["date"] = evaluate_df.forecast.apply(lambda x: x.index[0])
        evaluate_df.forecast = evaluate_df.forecast.apply(lambda x: x.values[0])
        mape, r2, rmse, mae, median_ape = calculate_performance_metrics(evaluate_df)
        _, text = residual_analysis(loops["loop_1"]["model"])
        return {"tests_data": evaluate_df, "metrics": {"mape": mape, "r2": r2, "rmse": rmse, "mae": mae, "mediana_mape": median_ape}, "analise_estatistica_dos_residuos_do_modelo":text}
    
    def forecast(self) -> DataFrame:
        self.model, forecast, predicts = self.setup_model(self.original_series, order=self.order, s_order=self.seasonal_order)
        return pd.concat([predicts.append(forecast), self.original_series], axis=1).rename(columns={0:"previsto","setor": "real"})

    def get_model_infos(self, index) -> Dict[str, Dict[any, any]]:
        model, forecast, predicts = self.setup_model(self.original_series.iloc[: index], order=self.order, s_order=self.seasonal_order)
        index = 0 if index is None else -index
        try:
            real = self.original_series.loc[forecast.index[0]].values[0]
        except:
            real = self.original_series.loc[forecast.index[0]]
        return {f"loop_{index}": {"model": model, "forecast": forecast, "predicts": predicts, "real": real}}

    def setup_model(self, series: Series, order: Tuple[int, int, int], s_order: Tuple[int, int, int, int]) -> Tuple[tsa.SARIMAX, Series, Series]:
        model = tsa.SARIMAX(series, order=order, seasonal_order=s_order)
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=1)
        predicts = fitted_model.predict()
        return model, forecast, predicts
    
    def show(self, title):
        df = self.forecast()
        df.plot(title=title, color=["#CE4446", "#113759"], figsize=(15,5))

    def test_model(self, n_loops:int=12) -> None:
        loops = {}
        for i in range(n_loops, 0, -1):
            loops |= self.get_model_infos(index=(None if i == 0 else -i))
        return loops
    
    def export_results(self, esp: str) -> Tuple[dict, pd.DataFrame]:
        metrics = prepare_metrics(self.evaluate_model())
        forecast = self.forecast()
        forecast.insert(2, "especialidade", esp)
        return metrics, forecast
