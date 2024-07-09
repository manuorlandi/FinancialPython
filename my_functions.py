import pandas as pd
from openbb_terminal.sdk import openbb
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix
import datetime
import sys, os
import sys
from datetime import date, datetime, timedelta
from multiprocessing import Pool
from os import environ, makedirs, remove
from os.path import exists
from pathlib import Path
from typing import Optional
from coinmetrics.api_client import CoinMetricsClient
from coinmetrics.constants import PagingFrom
import numpy as np
from os import path
from pathlib import Path
from binance.spot import Spot
from sklearn import metrics
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import itertools
import xgboost as xgb
import seaborn as sns
from typing import Union, Dict, List
from scipy import optimize
from scipy import special



def round_float_cols(
    data: pd.DataFrame, 
    howmanydecimals: int = 2
) -> pd.DataFrame:

    for col in data.columns:
        if(data[col].dtype == np.float64):
            data[col] = data[col].round(howmanydecimals)

    return data


def datetime_to_ts(date_time) -> int:

    if type(date_time)==datetime.date:
        return int(datetime.combine(date_time, datetime.time.min).timestamp()*1000)
    elif type(date_time)==datetime:
        return int(date_time.timestamp()*1000)

def ts_to_datetime(ts) -> str:
    
    return datetime.fromtimestamp(ts / 1000.0)

def download_data(client, pair, cols, PARAMS):

    client.klines(pair, interval=PARAMS['interval'], limit=1, startTime=PARAMS['startTime'])
    # Get server timestamp
    print(client.time())
    # Get klines of BTCUSDT at 1m interval
    start_timestamp = ts_to_datetime(client.klines(pair, interval=PARAMS['interval'], limit=1, startTime=PARAMS['startTime'])[0][0])
    end_timestamp = ts_to_datetime(PARAMS['endTime'])
    print(f'Exploring range {start_timestamp} - {end_timestamp}')

    df = pd.DataFrame()
    limit = 1000
    if end_timestamp <= start_timestamp: 

        print('No data are available for specified range')
    else:

        datediff = (end_timestamp - start_timestamp).days
        while datediff > 0:

            print(datediff)
            tmp_df = pd.DataFrame(client.klines(
                                            pair, 
                                            interval=PARAMS['interval'], 
                                            limit=limit, 
                                            startTime=datetime_to_ts(start_timestamp)
                                            )
            )

            start_timestamp += datetime.timedelta(days=limit)
            df = pd.concat([df, tmp_df], ignore_index=True)  
            datediff = (end_timestamp - start_timestamp).days
        
        df.columns = cols
    
    df['Symbol'] = pair
    df['OpenTimestamp'] = df['OpenTimestamp'].apply(ts_to_datetime)
    df['CloseTimestamp'] = df['CloseTimestamp'].apply(ts_to_datetime)

    dict_types= {
                'OpenTimestamp'                 :  'datetime64[ns]',
                'Open'                          :  'float64',
                'High'                          :  'float64',
                'Low'                           :  'float64',
                'Close'                         :  'float64',
                'Volume'                        :  'float64',
                'CloseTimestamp'                :  'datetime64[ns]',
                'Quote asset volume'            :  'float64',
                'Number of trades'              :  'int64',
                'Taker buy base asset volume'   :  'float64',
                'Taker buy quote asset volume'  :  'float64',
                'Ignore.'                       :  'int64',
                'Symbol'                        :  'str'
    }

    df= df.astype(dict_types)
    return df

def load_data(
            source: str,
            pair: str, 
            data_path, 
            PARAMS, 
            col_names, 
            overwrite: bool=False
) -> pd.DataFrame:

    file_name = pair.upper() + '_' + PARAMS['interval'] + '.csv'
    file_path = os.path.join(data_path,file_name)

    if not path.exists(file_path):
        print(f"File {file_name} not found.")
        df = download_data(source, pair, col_names, PARAMS)
        df.to_csv(file_path, index=False)
    else:
        if overwrite:
            print(f"Overfriting file {file_name}.")
            df = download_data(source, pair, col_names, PARAMS)
            df.to_csv(file_path, index=False)
        else:
            print(f"Loading {file_name}.")
            df = pd.read_csv(file_path)

    dict_types= {
                'OpenTimestamp'                 :  'datetime64[ns]',
                'Open'                          :  'float64',
                'High'                          :  'float64',
                'Low'                           :  'float64',
                'Close'                         :  'float64',
                'Volume'                        :  'float64',
                'CloseTimestamp'                :  'datetime64[ns]',
                'Quote asset volume'            :  'float64',
                'Number of trades'              :  'int64',
                'Taker buy base asset volume'   :  'float64',
                'Taker buy quote asset volume'  :  'float64',
                'Ignore.'                       :  'int64',
                'Symbol'                        :  'str'
    }
    
    df = df.astype(dict_types)
    return df

def get_top_symbols_by_marketcap(howmany, stable):
    
    mc = openbb.crypto.disc.coins(sortby = 'market_cap').head(howmany)
    tmc = mc['market_cap'].sum() 
    symbol_list = [element.upper() + stable for element in mc['symbol'] if 'USD' not in element.upper()]

    return symbol_list, tmc 

def get_all_pairs_available(client):

    list_pairs = client.exchange_info()
    pairs_available = [el['symbol'] for el in list_pairs['symbols'] if 'USDT' in el['symbol']]

    return pairs_available


def rolling_kpi(
    data: pd.DataFrame, 
    column: str , 
    window: int,
    method: str = 'avg', 
    return_df: bool = True
) -> pd.DataFrame:

    """
    Calcola la method rolling degli ultimi window periodi della colonna indicata
    """

    data_ = data.copy()    

    if str.lower(method) != 'ema':
        KPI = data_.rolling(window, min_periods=1).agg({column : method}).fillna(0).reset_index(drop=True)
    else:    
        KPI = data_[column].ewm(span=window).mean()

        
    if return_df:
        data_['rol_' + method + '_'+ column +'_'+str(window)] = KPI
        return data_
    else:
        return KPI


def plt_correlation(
    df: pd.DataFrame, 
    method:str='spearman'
):

    cols=list(df.columns) # list(df.columns[21:33])
    corr=df[cols].corr(method=method)
    corr

    # Generate a mask for the upper triangle
    mask = np.tril(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(20,20))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, vmax=1, center=0, 
                square=True, linewidths=1,  cbar_kws={"shrink": .5})



def cm_analysis(
    y_true, 
    y_pred, 
    title: str,
    filename: str, 
    labels: List[str], 
    ymap=None, 
    figsize=(10,10)
) -> None:
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.

    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      title:     plot name
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.

    Returns:
        None
    """

    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.2f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.2f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    cm_perc = pd.DataFrame(cm_perc, index=labels, columns=labels)
    cm_perc.index.name = 'Actual'
    cm_perc.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm_perc, annot=annot, fmt='', linewidths=1, ax=ax)
    plt.title('\n'+title+'\n', fontsize=14)
    plt.show()


def split_target_features(df: pd.DataFrame, feat_to_exclude=[], target_var='signal') ->List[pd.DataFrame]:
    """
    Given a DataFrame, split df into features and target 
    
     Args:
        df:         dataframe to be splitted
        target_var: variable target
    
    Returns:
        List[pd.DataFrame]: feature and target dataframe
    """
    features = df[[col for col in df.columns if col not in (feat_to_exclude + list(target_var))]].copy()
    labels = df[target_var].copy()
    
    return features, labels

def model_selection(
    model_type: str,
    parameter: Dict
) -> xgb.XGBClassifier:
    """
    Instantiate a tree based model
    
     Args:
        model_type: xgb/lgb currently supported
        parameter:  default parameters 
    
    Returns:
        Union[xgb.XGBClassifier, LGBMClassifier]: model instantiated
    """

    dict_model = {
            'xgb': xgb.XGBClassifier(n_jobs=-1, random_state=42),
            'rf': RandomForestClassifier(n_jobs=-1, random_state=42),
        }

    return dict_model[model_type].set_params(**parameter)
    
def dummy_train( 
    train_features: pd.DataFrame, 
    train_labels: pd.DataFrame, 
    val_features: pd.DataFrame, 
    val_labels: pd.DataFrame, 
    param: Dict, 
    n_est: int = 3000,
    model_type:str = 'xgb', 
    #metrics:List[callable] = cfg['TRAIN_VAL_METRICS'],
    verbosity: int = 50
) -> xgb.XGBClassifier:

    """
    Creates a tree based model and train it
    given training, validation set, parameters and model_type
    

    Args:
        train_features: train dataset without labels 
        train_labels:   train dataset labels
        val_features:   validation dataset without labels
        val_labels:     validation dataset labels
        param:          set of model parameters  
        n_est:          number of estimators
        model_type:     currently supported: xgb/lgb
        verbosity:      print every verbosity iterations
    
    Returns:
        xgb.XGBClassifier: model trained
    """


    model_base = model_selection(model_type, param)    
    model_ = model_base
    model_.set_params(n_estimators=n_est)
    print(model_.get_params())

    model_.fit(
        train_features, 
        train_labels,
        eval_set=[(train_features, train_labels), (val_features, val_labels)], 
        verbose=verbosity
    )
    return model_

def built_in_feature_importance(
    model: xgb.XGBClassifier, 
    feature_list: List[str],
    fixed_cols: List[str] = [],
    importance_type:str='weight',
    threshold: float = 0.001,
    plot=False
) -> Union[List[str], pd.DataFrame]:

    """
    Performs permutation importance taking advantage of built in methods

    Args:
        model_:          tree based model
        feature_list:    list of feature names
        fixed_cols:      columns that haven't to be removed
        importance_type: 'weight'/'gain'
        threshold:       value limit used to keep/drop a given feature
        plot:            True/False
    
    Returns:
        List[str]: list of feature to be dropped
        pd.DataFrame: dataframe containing permutation importance
    """

    model_ = model
    model_.importance_type = 'gain'
    feat_imp = model_.feature_importances_

    sorted_idx = feat_imp.argsort()
    # Ordino feature per valore decrescente di importanza
    data_perm_imp = pd.DataFrame({
            'features': feature_list[sorted_idx],
            'perm_imp': feat_imp[sorted_idx]
            })


    # Elimino le colonne con importanza negativa
    cols_to_remove = data_perm_imp.loc[data_perm_imp['perm_imp'] < threshold, 'features'].tolist()
    
    # Alcune colonne non devono essere rimosse
    cols_to_remove = [col for col in cols_to_remove if col not in fixed_cols]

    if plot:
        plt.figure(figsize=(12,18))
        plt.barh(data_perm_imp['features'], data_perm_imp['perm_imp'])
        plt.xlabel("Built-In Feature Importance ({})".format(importance_type))
    return cols_to_remove, data_perm_imp

def split_train_validation(
    df: pd.DataFrame, 
    tms,
    threshold: str
) -> List[pd.DataFrame]:
    """
    Split dataset in 2 datasets given threshold date

    args:
        df:        dataframe to be splitted
        threshold: date where dataframe will be cut

    Returns:
        List[pd.DataFrame]
    """    
    df_val = df[df[tms] > threshold].reset_index(drop=True)
    df_train = df[df[tms] <= threshold].reset_index(drop=True)
    #validation primo, train secondo
    
    return df_train, df_val

def return_index_if_exists(df_series, curr_idx, val, pos_crit, max_length):

    current_val = val
    if pos_crit:
        try:
            thing_index = list(el >= current_val for el in df_series[curr_idx+1:]).index(True) +1
        except ValueError:
            thing_index = max_length
    else:
        try:   
            thing_index = list(el <= current_val for el in df_series[curr_idx+1:]).index(True) +1
        except ValueError:
            thing_index = max_length

    return thing_index

def labelize_output_according_criterion(df: pd.DataFrame, target:list, wrt:str='Close', threshold:float=0.015, risk_reward_ratio:float=0.5, max_trade_length:int=5) -> pd.DataFrame:
    
    df_ = df.copy()
    positive_criterion = 1 + threshold/risk_reward_ratio
    negative_criterion = 1 - threshold
    
    CloseAbovethreshold = []
    CloseAbovethreshold = []
    HighAbovethreshold = []
    LowAbovethreshold = []
    OpenAbovethreshold = []
    CloseBelowthreshold = []
    HighBelowthreshold = []
    LowBelowthreshold = []
    OpenBelowthreshold = []

    for idx,row in df_.iterrows():
        #print(idx)
        pos_val = positive_criterion*row['Close']
        neg_val = negative_criterion*row['Close']
        if idx != df_.index[-1]:
            #get first value above/below threshold
            CloseAbovethreshold.append(return_index_if_exists(df_['Close'], idx, pos_val, True, max_trade_length))
            HighAbovethreshold.append(return_index_if_exists(df_['High'], idx, pos_val, True, max_trade_length))
            LowAbovethreshold.append(return_index_if_exists(df_['Low'], idx, pos_val, True, max_trade_length))
            OpenAbovethreshold.append(return_index_if_exists(df_['Open'], idx, pos_val, True, max_trade_length))
            CloseBelowthreshold.append(return_index_if_exists(df_['Close'], idx, neg_val, False, max_trade_length))
            HighBelowthreshold.append(return_index_if_exists(df_['High'], idx, neg_val, False, max_trade_length))
            LowBelowthreshold.append(return_index_if_exists(df_['Low'], idx, neg_val, False, max_trade_length))
            OpenBelowthreshold.append(return_index_if_exists(df_['Open'], idx, neg_val, False, max_trade_length))

    CloseAbovethreshold.append(0)
    HighAbovethreshold.append(0)
    LowAbovethreshold.append(0)
    OpenAbovethreshold.append(0)
    CloseBelowthreshold.append(0)
    HighBelowthreshold.append(0)
    LowBelowthreshold.append(0)
    OpenBelowthreshold.append(0)

    df_['CloseAbovethreshold'] = CloseAbovethreshold
    df_['HighAbovethreshold'] = HighAbovethreshold
    df_['LowAbovethreshold'] = LowAbovethreshold
    df_['OpenAbovethreshold'] = OpenAbovethreshold
    df_['CloseBelowthreshold'] = CloseBelowthreshold
    df_['HighBelowthreshold'] = HighBelowthreshold
    df_['LowBelowthreshold'] = LowBelowthreshold
    df_['OpenBelowthreshold'] = OpenBelowthreshold

    return df_


def labelize_output_according_criterion2(df: pd.DataFrame, map_ohlc:dict, wrt:str='Close', target_var:str='signal', threshold:float=0.015, risk_reward_ratio:float=.5, max_trade_length:int=5) -> pd.DataFrame:
    
    df_ = df.copy()
    positive_criterion = 1 + threshold/risk_reward_ratio
    negative_criterion = 1 - threshold
    
    min_above = []
    min_below = []

    df_['TP'] = df_[map_ohlc[wrt]]*positive_criterion
    df_['SL'] = df_[map_ohlc[wrt]]*negative_criterion

    for idx,row in df_.iterrows():

        pos_val = positive_criterion*row[map_ohlc[wrt]]
        neg_val = negative_criterion*row[map_ohlc[wrt]]

        if idx != df_.index[-1]:
            candidates_above_minima = []
            candidates_below_minima = []
            for _,v in map_ohlc.items():
                candidates_above_minima.append(return_index_if_exists(df_[v], idx, pos_val, True, max_trade_length))
                candidates_below_minima.append(return_index_if_exists(df_[v], idx, neg_val, False, max_trade_length))

            min_above.append(min(candidates_above_minima))
            min_below.append(min(candidates_below_minima))

    min_above.append(None)
    min_below.append(None)

    df_['min_above'] = min_above
    df_['min_below'] = min_below

    df_[target_var] = (df_['min_above'].lt(df_['min_below'])) & (df_['min_above'] <= max_trade_length)
    df_[target_var]=df_[target_var].astype(int)

    df_ = df_.drop(columns=['min_above','min_below'])

    return df_


def get_scores(
    model: xgb.XGBClassifier, 
    features: pd.DataFrame, 
    labels:pd.DataFrame
) -> pd.DataFrame:

    """
    Performs Auc, PR-Auc metrics on tree based 

    Args:
        model:  tree based model to evalute metrics above
        df:     target dataframe. Can be referred to train/validation/test dataset
        target: target
    
    Returns:
        pd.DataFrame: dataframe with results
    """

    
    preds = model.predict_proba(features)[:,1] 
    
    fpr, tpr, _ = metrics.roc_curve(labels, preds, pos_label=1)
    results = []

    print(np.round(metrics.auc(fpr, tpr),4))
    

    return results

def list_of_tuple_to_hline(list_tuple,df_series_ts):
    
    new_list = []
    for i in list_tuple:
        tmp = [(df_series_ts[i[0]],i[1]),(df_series_ts[len(df_series_ts)-1],i[1])]
        new_list.append(tmp)

    return new_list

#############################################################################################################################################################################
################################################################## SUPPORTS AND RESISTANCES #################################################################################
#############################################################################################################################################################################
# https://towardsdatascience.com/detection-of-price-support-and-resistance-levels-in-python-baedc44c34c9                                                                    #
# https://medium.datadriveninvestor.com/how-to-detect-support-resistance-levels-and-breakout-using-python-f8b5dac42f21                                                      #
#############################################################################################################################################################################
                                                                                                                                                                            #
# determine bullish fractal                                                                                                                                                 #
def is_support(df,i):                                                                                                                                                       #
  cond1 = df['Low'][i] < df['Low'][i-1]                                                                                                                                     #
  cond2 = df['Low'][i] < df['Low'][i+1]                                                                                                                                     #
  cond3 = df['Low'][i+1] < df['Low'][i+2]                                                                                                                                   #
  cond4 = df['Low'][i-1] < df['Low'][i-2]                                                                                                                                   #
  return (cond1 and cond2 and cond3 and cond4)                                                                                                                              #
                                                                                                                                                                            #
# determine bearish fractal                                                                                                                                                 #
def is_resistance(df,i):                                                                                                                                                    #
  cond1 = df['High'][i] > df['High'][i-1]                                                                                                                                   #
  cond2 = df['High'][i] > df['High'][i+1]                                                                                                                                   #
  cond3 = df['High'][i+1] > df['High'][i+2]                                                                                                                                 #
  cond4 = df['High'][i-1] > df['High'][i-2]                                                                                                                                 #
  return (cond1 and cond2 and cond3 and cond4)                                                                                                                              #
                                                                                                                                                                            #
# to make sure the new level area does not exist already                                                                                                                    #
def is_far_from_level(value, levels, df):                                                                                                                                   #
  ave =  np.mean(df['High'] - df['Low'])                                                                                                                                    #
  return np.sum([abs(value-level)<ave for _,level in levels])==0                                                                                                            #
                                                                                                                                                                            #
                                                                                                                                                                            #
                                                                                                                                                                            #
#############################################################################################################################################################################
# method 1: fractal candlestick pattern                                                                                                                                     #
#############################################################################################################################################################################
                                                                                                                                                                            #
def detect_level_method_1(df):                                                                                                                                              #
  levels = []                                                                                                                                                               #
  for i in range(2,df.shape[0]-2):                                                                                                                                          #
    if is_support(df,i):                                                                                                                                                    #
      l = df['Low'][i]                                                                                                                                                      #
      if is_far_from_level(l, levels, df):                                                                                                                                  #
        levels.append((i,l))                                                                                                                                                #
    elif is_resistance(df,i):                                                                                                                                               #
      l = df['High'][i]                                                                                                                                                     #
      if is_far_from_level(l, levels, df):                                                                                                                                  #
        levels.append((i,l))                                                                                                                                                #
  return levels                                                                                                                                                             #
                                                                                                                                                                            #
#############################################################################################################################################################################
# method 2: window shifting method                                                                                                                                          #
#############################################################################################################################################################################
                                                                                                                                                                            #
def detect_level_method_2(df):                                                                                                                                              #
  levels = []                                                                                                                                                               #
  max_list = []                                                                                                                                                             #
  min_list = []                                                                                                                                                             #
  for i in range(5, len(df)-5):                                                                                                                                             #
      high_range = df['High'][i-5:i+4]                                                                                                                                      #
      current_max = high_range.max()                                                                                                                                        #
      if current_max not in max_list:                                                                                                                                       #
          max_list = []                                                                                                                                                     #
      max_list.append(current_max)                                                                                                                                          #
      if len(max_list) == 5 and is_far_from_level(current_max, levels, df):                                                                                                 #
          levels.append((high_range.idxmax(), current_max))                                                                                                                 #
                                                                                                                                                                            #
      low_range = df['Low'][i-5:i+5]                                                                                                                                        #
      current_min = low_range.min()                                                                                                                                         #
      if current_min not in min_list:                                                                                                                                       #
          min_list = []                                                                                                                                                     #
      min_list.append(current_min)                                                                                                                                          #
      if len(min_list) == 5 and is_far_from_level(current_min, levels, df):                                                                                                 #
          levels.append((low_range.idxmin(), current_min))                                                                                                                  #
  return levels                                                                                                                                                             #
                                                                                                                                                                            #
#############################################################################################################################################################################
# to detect breakout                                                                                                                                                        #
#############################################################################################################################################################################
                                                                                                                                                                            #
def has_breakout(levels, previous, last):                                                                                                                                   #
  for _, level in levels:                                                                                                                                                   #
    cond1 = (previous['Open'] < level)                                                                                                                                      #
    cond2 = (last['Open'] > level) and (last['Low'] > level)                                                                                                                #
  return (cond1 and cond2)                                                                                                                                                  #
                                                                                                                                                                            #
#############################################################################################################################################################################
#############################################################################################################################################################################
#############################################################################################################################################################################


def plt_kmeans(X):

    # Set a range for the number of clusters
    clusters = range(1, 10)

    # Create an empty list to store the inertia values
    inertias = []

    # Fit and calculate inertia for each number of clusters
    for k in clusters:
        model = KMeans(n_clusters=k)
        model.fit(X)
        inertias.append(model.inertia_)

    # Plot the inertia values
    plt.plot(clusters, inertias, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

def plt_silhouette(X):
    # Set a range for the number of clusters
    clusters = range(2, 10)

    # Create an empty list to store the silhouette scores
    scores = []

    # Fit and calculate silhouette score for each number of clusters
    for k in clusters:
        model = KMeans(n_clusters=k)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append(score)

    # Plot the silhouette scores
    plt.plot(clusters, scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.show()



#############################################################################################################################################################################
def robust_pow(num_base, num_pow):
    # numpy does not permit negative numbers to fractional power
    # use this to perform the power algorithmic

    return np.sign(num_base) * (np.abs(num_base)) ** (num_pow)

#############################################################################################################################################################################
# CUSTOM  LOSS ##############################################################################################################################################################
#############################################################################################################################################################################

# FOCAL LOSS

class FocalLoss:

    def __init__(self, gamma, alpha=None):
        self.alpha = alpha
        self.gamma = gamma

    def __getstate__(self):
        #print("I'm being pickled")
        return self.__dict__
        
    def __setstate__(self, d):
        #print("I'm being unpickled with these values: " + d)
        self.__dict__ = d

    def at(self, y):
        if self.alpha is None:
            return np.ones_like(y)
        return np.where(y, self.alpha, 1 - self.alpha)

    def pt(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.where(y, p, 1 - p)

    def __call__(self, y_true, y_pred):
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        return -at * (1 - pt) ** self.gamma * np.log(pt)

    def grad(self, y_true, y_pred):
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma
        return at * y * (1 - pt) ** g * (g * pt * np.log(pt) + pt - 1)

    def hess(self, y_true, y_pred):
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma

        u = at * y * (1 - pt) ** g
        du = -at * y * g * (1 - pt) ** (g - 1)
        v = g * pt * np.log(pt) + pt - 1
        dv = g * np.log(pt) + g + 1

        return (du * v + u * dv) * y * (pt * (1 - pt))

    def init_score(self, y_true):
        res = optimize.minimize_scalar(
            lambda p: self(y_true, p).sum(),
            bounds=(0, 1),
            method='bounded'
        )
        p = res.x
        log_odds = np.log(p / (1 - p))
        return log_odds

    def lgb_obj(self, y,preds):
        p = special.expit(preds)
        return self.grad(y, p), self.hess(y, p)

    def lgb_eval(self, preds, y):
        p = special.expit(preds)
        is_higher_better = False
        return 'focal_loss', self(y, p).mean(), is_higher_better


#############################################################################################################################################################################
# GET DATA FROM ALL THE EXHANGES ############################################################################################################################################
#############################################################################################################################################################################
                                                                                                                                                                            #
def export_data(client,                                                                                                                                                     #
                PROCESSED_DAYS_REGISTRY_FILE_PATH,                                                                                                                          #
                EXCHANGES_TO_EXPORT,                                                                                                                                        #
                MARKETS_TO_EXPORT,                                                                                                                                          #
                MARKET_TYPES_TO_COLLECT,                                                                                                                                    #
                BASE_MARKETS,                                                                                                                                               #
                QUOTE_MARKETS,                                                                                                                                              #
                EXPORT_START_DATE,                                                                                                                                          #
                EXPORT_END_DATE,                                                                                                                                            #
                DST_ROOT,                                                                                                                                                   #
                FREQUENCY                                                                                                                                                   #
):                                                                                                                                                                          #
                                                                                                                                                                            #
    min_export_date = date.fromisoformat(EXPORT_START_DATE)                                                                                                                 #
    max_export_date = (                                                                                                                                                     #
        date.fromisoformat(EXPORT_END_DATE)                                                                                                                                 #
        if EXPORT_END_DATE is not None                                                                                                                                      #
        else date.today() - timedelta(days=1)                                                                                                                               #
    )                                                                                                                                                                       #
    processed_dates_and_markets = read_already_processed_files(PROCESSED_DAYS_REGISTRY_FILE_PATH)                                                                           #
                                                                                                                                                                            #
    markets = get_markets_to_process(client,EXCHANGES_TO_EXPORT, MARKETS_TO_EXPORT, MARKET_TYPES_TO_COLLECT, BASE_MARKETS, QUOTE_MARKETS)                                   #
    print("getting markets: %s", [market["market"] for market in markets])                                                                                                  #
    processes_count = 2                                                                                                                                                     #
                                                                                                                                                                            #
    if processes_count > 2:                                                                                                                                                 #
        print(                                                                                                                                                              #
            "Using more than two parallel processes will likely not result into faster export."                                                                             #
        )                                                                                                                                                                   #
                                                                                                                                                                            #
    with Pool(processes_count) as pool:                                                                                                                                     #
        tasks = []                                                                                                                                                          #
        for market in markets:                                                                                                                                              #
            market_data_root = "/".join(                                                                                                                                    #
                (                                                                                                                                                           #
                    DST_ROOT.rstrip("/"),                                                                                                                                   #
                    market["market"].split("-")[0],                                                                                                                         #
                    get_instrument_root(market),                                                                                                                            #
            )                                                                                                                                                               #
                )                                                                                                                                                           #
                                                                                                                                                                            #
            # creating all the directories upfront to not to call this function in export function for each day                                                             #
            # otherwise it can fail in the multiproc environment even with exist_ok=True.                                                                                   #
            makedirs(market_data_root, exist_ok=True)                                                                                                                       #
                                                                                                                                                                            #
            if FREQUENCY in {"1h", "4h", "1d"}:                                                                                                                             #
                if (                                                                                                                                                        #
                                                                                                                                                                            #
                    get_registry_key(market, min_export_date)                                                                                                               #
                    not in processed_dates_and_markets                                                                                                                      #
                ):                                                                                                                                                          #
                    tasks.append(                                                                                                                                           #
                            export_data_for_a_market                                                                                                                        #
                            (                                                                                                                                               #
                                client,                                                                                                                                     #
                                market,                                                                                                                                     #
                                market_data_root,                                                                                                                           #
                                min_export_date,                                                                                                                            #
                                max_export_date,                                                                                                                            #
                                FREQUENCY, PROCESSED_DAYS_REGISTRY_FILE_PATH                                                                                                #
                            )                                                                                                                                               #
                                                                                                                                                                            #
                    )                                                                                                                                                       #
            else:                                                                                                                                                           #
                for target_date in get_days_to_export(                                                                                                                      #
                    market, min_export_date, max_export_date                                                                                                                #
                ):                                                                                                                                                          #
                    if (                                                                                                                                                    #
                        get_registry_key(market, target_date)                                                                                                               #
                        not in processed_dates_and_markets                                                                                                                  #
                    ):                                                                                                                                                      #
                        tasks.append(                                                                                                                                       #
                            pool.apply_async(                                                                                                                               #
                                export_data_for_a_market,                                                                                                                   #
                                (client,market, market_data_root, target_date, target_date,FREQUENCY, PROCESSED_DAYS_REGISTRY_FILE_PATH),                                   #
                            )                                                                                                                                               #
                        )                                                                                                                                                   #
                                                                                                                                                                            #
        start_time = datetime.utcnow()                                                                                                                                      #
                                                                                                                                                                            #
        for i, task in enumerate(tasks, 1):                                                                                                                                 #
            try:                                                                                                                                                            #
                task.get()                                                                                                                                                  #
            except Exception:                                                                                                                                               #
                print(f'failed to get data for task {task}')                                                                                                                #
            time_since_start = datetime.utcnow() - start_time                                                                                                               #
            print("processed task: %s/%s, time since start: %s, completion ETA:: %s",                                                                                       #
                        i, len(tasks), time_since_start, time_since_start / i * (len(tasks) - i))                                                                           #
                                                                                                                                                                            #
                                                                                                                                                                            #
#############################################################################################################################################################################
                                                                                                                                                                            #
def get_instrument_root(market):                                                                                                                                            #
    if market["type"] == "spot":                                                                                                                                            #
        return "{}_{}_{}".format(market["base"], market["quote"], market["type"])                                                                                           #
    return "{}_{}".format(market["symbol"].replace(":", "_"), market["type"])                                                                                               #
                                                                                                                                                                            #
#############################################################################################################################################################################
                                                                                                                                                                            #
def read_already_processed_files(PROCESSED_DAYS_REGISTRY_FILE_PATH):                                                                                                        #
    if exists(PROCESSED_DAYS_REGISTRY_FILE_PATH):                                                                                                                           #
        with open(PROCESSED_DAYS_REGISTRY_FILE_PATH) as registry_file:                                                                                                      #
            return set(registry_file.read().splitlines())                                                                                                                   #
    return set()                                                                                                                                                            #
                                                                                                                                                                            #
#############################################################################################################################################################################
                                                                                                                                                                            #
def get_markets_to_process(client,EXCHANGES_TO_EXPORT, MARKETS_TO_EXPORT, MARKET_TYPES_TO_COLLECT, BASE_MARKETS, QUOTE_MARKETS):                                            #
    markets = []                                                                                                                                                            #
                                                                                                                                                                            #
    for exchange in EXCHANGES_TO_EXPORT or [None]:                                                                                                                          #
        for market in client.catalog_markets(exchange=exchange):                                                                                                            #
            if market["market"] in MARKETS_TO_EXPORT or (                                                                                                                   #
                (market["type"] in MARKET_TYPES_TO_COLLECT)                                                                                                                 #
                and (                                                                                                                                                       #
                    (                                                                                                                                                       #
                        ('base' in market and market["base"] in BASE_MARKETS or not BASE_MARKETS)                                                                           #
                        and (market["quote"] in QUOTE_MARKETS or not QUOTE_MARKETS)                                                                                         #
                    )                                                                                                                                                       #
                )                                                                                                                                                           #
            ):                                                                                                                                                              #
                markets.append(market)                                                                                                                                      #
    return markets                                                                                                                                                          #
                                                                                                                                                                            #
#############################################################################################################################################################################
                                                                                                                                                                            #
def get_days_to_export(market_info, min_export_date, max_export_date):                                                                                                      #
    min_date = max(                                                                                                                                                         #
        date.fromisoformat(market_info["min_time"].split("T")[0]), min_export_date                                                                                          #
    )                                                                                                                                                                       #
    max_date = min(                                                                                                                                                         #
        date.fromisoformat(market_info["max_time"].split("T")[0]), max_export_date                                                                                          #
    )                                                                                                                                                                       #
    for target_date_index in range((max_date - min_date).days + 1):                                                                                                         #
        yield min_date + timedelta(days=target_date_index)                                                                                                                  #
                                                                                                                                                                            #
#############################################################################################################################################################################
                                                                                                                                                                            #
def export_data_for_a_market(client,market, market_data_root, start_date, end_date, FREQUENCY, PROCESSED_DAYS_REGISTRY_FILE_PATH):                                          #
    market_candles = client.get_market_candles(                                                                                                                             #
        market["market"],                                                                                                                                                   #
        start_time=start_date,                                                                                                                                              #
        end_time=end_date,                                                                                                                                                  #
        page_size=10000,                                                                                                                                                    #
        paging_from=PagingFrom.START,                                                                                                                                       #
        frequency=FREQUENCY,                                                                                                                                                #
    )                                                                                                                                                                       #
    if start_date != end_date:                                                                                                                                              #
        dst_csv_file_path = "/".join((market_data_root, "candles")) + f"{FREQUENCY}.csv"                                                                                    #
    else:                                                                                                                                                                   #
        dst_csv_file_path = (                                                                                                                                               #
            "/".join((market_data_root, "candles_" + start_date.isoformat())) + f"{FREQUENCY}.csv"                                                                          #
        )                                                                                                                                                                   #
                                                                                                                                                                            #
    print("downloading data to: %s", dst_csv_file_path)                                                                                                                     #
    market_candles.export_to_csv(dst_csv_file_path)                                                                                                                         #
    # cleanup files without data                                                                                                                                            #
    if Path(dst_csv_file_path).stat().st_size == 0:                                                                                                                         #
        remove(dst_csv_file_path)                                                                                                                                           #
    with open(PROCESSED_DAYS_REGISTRY_FILE_PATH, "a") as registry_file:                                                                                                     #
        registry_file.write(get_registry_key(market, start_date) + "\n")                                                                                                    #
                                                                                                                                                                            #
#############################################################################################################################################################################
                                                                                                                                                                            #
def get_registry_key(market, target_date):                                                                                                                                  #
    return "{},{}".format(market["market"], target_date.isoformat())                                                                                                        #
                                                                                                                                                                            #
#############################################################################################################################################################################

def list_of_all_metrics(client):
    
    metric = []
    full_name = []
    desc = []

    list_metrics = client.catalog_asset_metrics()
    for el in list_metrics:
        metric.append(el['metric'])
        full_name.append(el['full_name'])
        desc.append(el['description'])

    df = pd.DataFrame({'metric':metric, 'full_name':full_name, 'description':desc})
    return df

#############################################################################################################################################################################


def read_data_for_a_market(market_data_root, FREQUENCY):                                                 
                                                                                                                                                               
    dst_csv_file_path = "/".join((market_data_root, "candles")) + f"{FREQUENCY}.csv"                                                                                                
                                                                                                                                                           
    print("reading data from: ", dst_csv_file_path)                                                                                                               
    tmp = pd.read_csv(dst_csv_file_path)                                                                                                                         
    return tmp      