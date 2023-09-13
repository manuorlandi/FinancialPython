from os import path
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from twelvedata import TDClient
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf
from pathlib import Path
from prophet import Prophet
from fbprophet.diagnostics import cross_validation,performance_metrics
from prophet.plot import plot_plotly, plot_components_plotly
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import itertools

apikey="7574f9ef4be7496e976c0d1652415f34"

def get_historicaldata(mykey, symbol, interval, outputsize,timezone) -> TDClient:

    # Initialize client - apikey parameter is requiered
    td = TDClient(apikey=mykey)
    # Construct the necessary time serie
    ts = td.time_series(
        symbol=symbol,
        interval=interval,
        outputsize=outputsize,
        timezone=timezone,
    )

    return ts.with_bbands(ma_type="EMA").with_plus_di().with_wma(time_period=20).with_wma(time_period=40)
    #return ts.as_pandas()

def load_data(apikey:str, project_path:str, symbol: str, interval: str, outputsize: str, timezone: str, n: int, overwrite: bool=False) -> DataFrame:
    
    data_path = project_path / (symbol.replace("/","_").upper() + '_' + interval + '.csv')

    if not path.exists(data_path):
        df = get_historicaldata(apikey, symbol, interval, outputsize, timezone).as_pandas()
        df.to_csv(data_path)
    else:
        if overwrite:
            df = get_historicaldata(apikey, symbol, interval, outputsize, timezone).as_pandas()
            df.to_csv(data_path)
        else:
            df = pd.read_csv(data_path, parse_dates=True, nrows=n) #index_col=0
    
    print(df.head(100))
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(df.head(100))
    df.sort_values(by='datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def plot_series(df_: DataFrame, simple=True):
    #set datetime (later Date) as index in order to prepare the data to be plotted and sort df by date ascending

    df = df_.copy()

    if simple:
        #sns.lineplot(x="datetime", y="open",legend = 'full' , data=df_[['datetime','open']])
        sns.lineplot(x="datetime", y="open",legend = 'full' , data=df[['datetime','open']]) #[:28])
        plt.show()
        sns.lineplot(x="datetime", y="open",legend = 'full' , data=df[['datetime','open']][:28])
        plt.show()
        df['weekday'] = df['datetime'].dt.weekday  
        sns.boxplot(x="weekday", y="open", data=df)
        plt.show()
        df.set_index('datetime', inplace=True)
        result = seasonal_decompose(df['open'], model='additive', period=365)
        fig = result.plot()  
        fig.set_size_inches(12, 9)
        return

    df.set_index('datetime', inplace=True)
    df.index.name = 'Date'
    df.sort_index(ascending=True, inplace=True)

    apdict = mpf.make_addplot(df[['upper_band','middle_band','lower_band','plus_di','wma1','wma2']])

    apds = [ mpf.make_addplot(df['upper_band'], color='b'),
         mpf.make_addplot(df['lower_band'], color='b'),
         mpf.make_addplot(df['wma1'],color='g'),
         mpf.make_addplot((df['wma2']),color='y')
       ]

    mc = mpf.make_marketcolors(up='black',down='black', edge={'up':'g','down':'r'}, wick='white')
    s  = mpf.make_mpf_style(base_mpf_style='nightclouds')#, marketcolors=mc)
    mpf.plot(df,type='candle', addplot=apds,style=s)

def days_of_week(ds: pd.Series) -> pd.Series:
    date = pd.to_datetime(ds)
    if not(not(date.weekday()) == 5 and not(date.weekday()) == 6):
        return 1
    else:
        return 0

def plot_acf_pacf(df: pd.DataFrame, field:str, lags: int) -> None:

    decomp = df[['ds',field]] 
    decomp.set_index('ds', inplace=True)

    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = plot_acf(decomp, lags=lags, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = plot_pacf(decomp, lags=lags, ax=ax2)

    plt.show()

def test_stationarity(timeseries, window = 12, cutoff = 0.01):

    #Determing rolling statistics
    rolmean = timeseries.rolling(window).mean()
    rolstd = timeseries.rolling(window).std()

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC', maxlag = 20 )
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    pvalue = dftest[1]
    if pvalue < cutoff:
        print('p-value = %.4f. The series is likely stationary.' % pvalue)
    else:
        print('p-value = %.4f. The series is likely non-stationary.' % pvalue)
    
    print(dfoutput)


PROJECT_DIR = Path(Path.cwd())
DATA_FOLDER = PROJECT_DIR / 'data/'

symbol="BTC/USD"
interval="1h"
outputsize=5000
timezone="America/New_York"
howmanyrows=5000
n_test=50

plot = False

df = load_data(apikey, DATA_FOLDER, symbol, interval, outputsize, timezone, howmanyrows)
print(df.head(10))

#test_stationarity(df['open'])

'''print(decomp.dtypes)
print(decomp.head(10))
result = seasonal_decompose(decomp, model='additive', period=52)

result.plot()
'''

#plt.show()

df.rename(columns={"datetime": "ds", "open": "y"}, inplace=True)
X = df[df.columns.drop('y')]
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


train_size = X_train.shape[0]
test_size = X_test.shape[0]

df_test_orig = df[['ds', 'y']]
df_train = df_test_orig[:train_size]
df_test = df_test_orig[:train_size+1]
#test_stationarity(df['open_diff'], window = 12)

#mpf.plot(df,type='candle', addplot=apds,style='mike')

if plot:
    #plot_series(df)
    
    df_train = df_test[:howmanyrows-n_test]
    df_test = df_test[howmanyrows-n_test:]

    df_test['open_diff'] = df['y'] - df['y'].shift(1)
    df_test['open_diff'] = df_test['open_diff'].fillna(0)
    plot_acf_pacf(df_test[:50], 'open_diff', lags=10)


print('*************************************** aaaaaaaaaaaaaaaaa *************************************')
print('*************************************** aaaaaaaaaaaaaaaaa *************************************')
print('*************************************** aaaaaaaaaaaaaaaaa *************************************')
print(df.columns)

## Creating model parameters


param_grid={
    "daily_seasonality": [False],
    "weekly_seasonality":[False],
    "yearly_seasonality":[False],
    "growth": ["logistic"],
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5], # to give higher value to prior trend
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0] # to control the flexibility of seasonality components
}
# Generate all combination of parameters
all_params= [
            dict(zip(param_grid.keys(), v))
            for v in itertools.product(*param_grid.values())
]

rmses= list ()

for model_param in all_params:
    m = Prophet(**model_param)
    m = m.add_seasonality(name="monthly", period=30, fourier_order=10)
    m = m.add_seasonality(name="quarterly", period=92.25, fourier_order=10)
    df_train['cap']= df_train["y"].max() + df_train["y"].std() * 0.05 
    m.fit(df_train)
    
    df_cv= cross_validation(m, initial="336 hours", period="168 hours", horizon="168 hours")
    df_p= performance_metrics(df_cv, rolling_window=1)
    rmses.append(df_p['rmse'].values[0])

# find teh best parameters
best_params = all_params[np.argmin(rmses)]
                            
print("\n The best parameters are:", best_params)

'''m.add_regressor('open')
m.add_regressor('high')
m.add_regressor('low')
m.add_regressor('volume')
m.add_regressor('upper_band')
m.add_regressor('middle_band')
m.add_regressor('lower_band')
m.add_regressor('plus_di')
m.add_regressor('wma1')
m.add_regressor('wma2')
'''

future = m.make_future_dataframe(periods=test_size, freq='H')
future['cap'] = df_train['cap'].max()
#future = future.fillna(0)
print(future.tail())

forecast = m.predict(future)
print('*************************************** FUTURE *************************************')
print('*************************************** FUTURE *************************************')
print('*************************************** FUTURE *************************************')


f, ax = plt.subplots(1)
fig = m.plot(forecast, ax=ax)
ax.scatter(df_test_orig['ds'], df_test_orig['y'], color='r',s=0.4)
fig = m.plot(forecast, ax=ax)
plt.show()
m.plot_components(forecast)
plt.show()
