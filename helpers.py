
import holidays
import numpy as np
import pandas as pd 
from tqdm import tqdm
from datetime import date
from prophet import Prophet
from itertools import product
import matplotlib.pyplot as plt 
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error as MAE 
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from sklearn.metrics import mean_absolute_percentage_error as MAPE 
from prophet.diagnostics import cross_validation,performance_metrics

PC = '#FF3B2B'



def tsplot(y, lags=None, figsize=(12, 7), style="bmh"):
    """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test

        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))

        ts_ax.plot(y,color='black',alpha=0.6)
        p_value = adfuller(y)[1]
        ts_ax.set_title(
            "Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}".format(p_value),
            fontsize=17,fontweight='semibold',fontfamily='serif'
        )
        plot_acf(y, lags=lags, ax=acf_ax,color=PC)
        plot_pacf(y, lags=lags, ax=pacf_ax,color=PC)
        plt.tight_layout()

def optimizeARIMA(data,parameters_list, d):
    """
        Return dataframe with parameters and corresponding AIC

        parameters_list - list with (p, q, P, Q) tuples
        d - integration order in ARIMA model
        D - seasonal integration order
        s - length of season
    """

    results = []
    best_aic = float("inf")

    for param in tqdm(parameters_list):
        # we need try-except because on some combinations model fails to converge
        try:
            model = SARIMAX(
                data,
                order=(param[0], d, param[1]),
            ).fit(disp=-1)
        except:
            continue
        aic = model.aic
        # saving best model, AIC and parameters
        if aic < best_aic:
            best_model = model
            best_aic = aic
        results.append([param, model.aic])

    result_table = pd.DataFrame(results)
    result_table.columns = ["parameters", "aic"]
    # sorting in ascending order, the lower AIC is - the better
    result_table = result_table.sort_values(by="aic", ascending=True).reset_index(
        drop=True
    )

    return result_table,best_model

def optimizeSARIMAX(data,parameters_list, d, D, s):
    """
        Return dataframe with parameters and corresponding AIC

        parameters_list - list with (p, q, P, Q) tuples
        d - integration order in ARIMA model
        D - seasonal integration order
        s - length of season
    """

    results = []
    best_aic = float("inf")

    for param in tqdm(parameters_list):
        # we need try-except because on some combinations model fails to converge
        try:
            model = SARIMAX(
                data,
                order=(param[0], d, param[1]),
                seasonal_order=(param[2], D, param[3], s),
            ).fit(disp=-1)
        except:
            continue
        aic = model.aic
        # saving best model, AIC and parameters
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])

    result_table = pd.DataFrame(results)
    result_table.columns = ["parameters", "aic"]
    # sorting in ascending order, the lower AIC is - the better
    result_table = result_table.sort_values(by="aic", ascending=True).reset_index(
        drop=True
    )

    return result_table

def get_country_holydays(countries,start_year,end_year,weights=None):
    start_date = date(start_year,1,1)
    end_date = date(end_year,12,31)

    holidays_df = pd.DataFrame(data=pd.date_range(start_date,end_date),columns=['date'])
    for country in countries:
        try:
            h = holidays.country_holidays(country)
            holidays_df[country] = holidays_df['date'].apply(lambda x:int(x in h))
            if weights:holidays_df[country] = holidays_df[country].multiply(weights[country])
        except NotImplementedError:
            pass
    return holidays_df.set_index('date').sum(axis=1)

def is_weekend(df):
    df = df.copy()
    df['weekend'] = [x.day_name() for x in df.index]
    df['weekend'] = df['weekend'].apply(lambda x:1 if x in['Saturday','Sunday'] else 0)
    return df

def plot_predictions(observed,predicted,metric='MAPE'):
    if metric=='MAPE':
        error = "MAPE: {0:.3f}%".format(MAPE(observed,predicted)*100)
    else:
        error = "MAE: {0:.3f}".format(MAE(observed,predicted))

    plt.figure(figsize=(15,7))
    plt.plot(observed,color='black',label='oberved')
    plt.plot(predicted,color='#FF3B2B',label='predicted')
    plt.legend(loc='upper center')
    plt.xlabel("Date",fontfamily='serif')
    plt.ylabel("Views (Mn)",fontfamily='serif')
    plt.title(f"Predictions vs Observed\n{error}", fontsize=18,fontweight='semibold',fontfamily='serif')
    plt.grid(True)
    plt.show()

def plotSARIMA(model,train,test):
    predictions = model.get_prediction(train.shape[0],train.shape[0]+99)
    predictions_conf_int = predictions.conf_int()
    predictions = pd.DataFrame(index=test.index,data=predictions.predicted_mean.values)
    
    plt.figure(figsize=(15,7))
    plt.plot(test,color='black',label='oberved')
    plt.plot(predictions,color='#FF3B2B',label='predicted')
    plt.fill_between(predictions.index,
                    predictions_conf_int.iloc[:, 0],
                    predictions_conf_int.iloc[:, 1], color='g', alpha=0.3)
    plt.legend(loc='upper center')
    plt.xlabel("Date",fontfamily='serif')
    plt.ylabel("Views (Mn)",fontfamily='serif')
    plt.title(f"Predictions vs Observed\n MAPE:{MAPE(test,predictions)*100:.3f}%", fontsize=18,fontweight='semibold',fontfamily='serif')
    plt.grid(True)
    plt.show()

def optimizeProphet(param_grid,train,initial,period,horizon):
    args = list(product(*param_grid.values()))
    df_ps = pd.DataFrame()
    for arg in tqdm(args):
        kwargs = dict(zip(param_grid.keys(),arg))
        m = Prophet(**kwargs).fit(train)
        df_cv = cross_validation(m, initial=initial, period=period, horizon = horizon)
        df_p = performance_metrics(df_cv, rolling_window=1)
        df_p['params'] = str(arg)
        df_ps = df_ps.append(df_p)
    df_ps['mae+rmse'] = df_ps['mae']+df_ps['rmse']
    df_ps = df_ps.sort_values(['mae+rmse'])
    return df_ps,df_p,df_cv

def plot_ts_cluster_results(usage,j,n,plot_mean=False):
    plt.figure(figsize=(15,7))
    if plot_mean:mean_usage=np.array(usage).transpose().mean(axis=1)
    for x,i in zip(usage,np.linspace(0.1,1,n)):
        plt.plot(x,alpha=i,linewidth=3,color='#007E9A')

    if plot_mean:plt.plot(mean_usage,alpha=0.9,linewidth=3,color='k',linestyle='--')

    plt.title(f'Clustering results: Cluster 0{j+1}',fontfamily='serif',fontsize='17',fontweight='semibold')
    plt.show()