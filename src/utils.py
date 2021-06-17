import pandas as pd
import numpy as np
from fredapi import Fred
import yaml
import wrds
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
from sklearn.decomposition import PCA
import keras
from keras import layers, models
from tensorflow import keras
from sklearn.metrics import mean_squared_error
import pickle as pkl
import os

def round_datetime(date):
    """round a given date to the first of the current month"""
    return pd.to_datetime(str(date)[:7])

def collect_financial_data(DATA_DIR, db):
    """
    Collects financial data upon request, expects wrds connection
    :param DATA_DIR: string, Data directory location
    :param db: wrds database connection object
    :return: None, financial data is saved under provided DATA_DIR
    """
    
    all_common_stocks = db.raw_sql("SELECT a.permno, a.date, b.shrcd, b.exchcd, a.ret, a.prc, a.shrout"
                                  " FROM crsp.dsf AS a LEFT JOIN crsp.msenames as b"
                                  " ON a.permno=b.permno AND b.namedt<=a.date AND a.date<=b.nameendt"
                                  " WHERE b.shrcd in (10,11) AND b.exchcd in (1,2)"
                                  " AND a.date >='1992-01-01' AND a.date <='2019-12-31'"
                                  )
    all_common_stocks['market_cap'] = np.abs(all_common_stocks['prc']) * all_common_stocks['shrout']
    all_common_stocks['year_month'] = pd.to_datetime(all_common_stocks['date']).dt.to_period('M')
    all_common_stocks = all_common_stocks.groupby(['permno','year_month']).agg(['mean', 'std']).reset_index()[['year_month','permno','ret','market_cap']]
    all_common_stocks.columns = [f'{i}{j}' for i,j in all_common_stocks.columns.to_flat_index()]
    all_common_stocks['date'] = pd.to_datetime(all_common_stocks['year_month'].dt.to_timestamp(), format='%Y-%m')

    # top 20 market-cap stocks end of 2019
    top20 = all_common_stocks.loc[all_common_stocks['date']=='2019-12-01'].sort_values('market_capmean', ascending=False)[:10].permno
    top20_ts = all_common_stocks.loc[all_common_stocks.permno.isin(top20)].pivot_table(values=['retmean','retstd'], columns='permno', index='date')
    top20_ts.columns = [f'{i}{j}' for i,j in top20_ts.columns.to_flat_index()]

    bonds=db.raw_sql("select caldt, b2ret, b10ret from crsp.mcti where caldt between '1992-12-31' and '2020-12-31'", date_cols=['caldt'])
    bonds=bonds.rename(columns= {"caldt": "date"})
    bonds['date'] = bonds['date'].map(round_datetime)
    bonds = bonds.set_index('date')

    rf = db.raw_sql("select mcaldt, tmytm from crsp.tfz_mth_rf where kytreasnox = 2000001 and mcaldt>='1992-12-31' and mcaldt<='2020-12-31'")
    rf['mcaldt'] = rf['mcaldt'].map(round_datetime)
    rf = rf.set_index('mcaldt')

    msi = pd.read_csv(DATA_DIR + "Historical Major stock indexs.csv")
    msi['Date'] = pd.to_datetime(msi['Date'], format='%d/%m/%Y')
    msi = msi[["Index","Date","Adj"]].pivot_table(values='Adj', columns='Index', index='Date')
    msi = msi.pct_change(periods=1).dropna()

    financial_data = bonds.merge(rf, left_index=True, right_index=True).merge(msi, left_index=True, right_index=True).merge(top20_ts, left_index=True, right_index=True).dropna(axis=1)

    financial_data.reset_index().to_csv(DATA_DIR + "financial data.csv", index = False)

def clean_release_data(fred, indicator, impute_method, lag='release'):
    """
    Collects Macroeconomic data and performs imputation, supported are 'sarimax','linear interpolation' and 'backfill'
    :param fred: fred api connection object
    :param indicator: string, name of desired (according to fredapi database) macroeconomic serie 
    :param impute_method: imputation method to use, supported are 'sarimax','linear interpolation' and 'backfill'
    :return: None, financial data is saved under provided DATA_DIR
    """
    
    lag = 'realtime_start' if lag == 'release' else 'date'
    temp = fred.get_series_all_releases(indicator).sort_values('realtime_start')
    temp = temp[~temp.date.dt.date.duplicated()]
    temp = temp.sort_values('date')
    temp = temp[~temp.realtime_start.dt.date.duplicated()]
    temp = temp.loc[temp.realtime_start > '1992-12-01']
    temp[lag] = temp[lag].map(round_datetime)
    temp = temp.set_index(lag)

    if impute_method == 'sarimax':
        temp = temp['value'].pct_change(periods=1).dropna().resample('M').first().to_frame('value')
        data = np.asarray(temp.value).astype('float')

        # Iterate over all ARMA(p,q) models with p,q in [0,6]
        best_p, best_q = 0,0
        best_aic = np.inf
        for p in range(6):
            for q in range(6):
                if p == 0 and q == 0:
                    continue

                # Estimate the model with missing datapoints
                mod = sm.tsa.statespace.SARIMAX(data, order=(p,0,q), enforce_invertibility=False)
                try:
                    res = mod.fit(disp=False)
                    if best_aic > res.aic and res.aic>0: 
                        best_p, best_q, best_aic = p, q, res.aic
                except:
                    continue

        mod = sm.tsa.statespace.SARIMAX(data, order=(best_p,0,best_q), enforce_invertibility=False)
        res = mod.fit(disp=False)

        # In-sample one-step-ahead predictions
        predict = res.get_prediction(end=mod.nobs).predicted_mean
        data[np.where(np.isnan(data))[0]] = predict[np.where(np.isnan(data))[0]].tolist()
        data[0] = np.nan
        temp['value'] = data[0:data.shape[0]]
        temp = temp['value'].dropna()
        temp.index = temp.index.map(round_datetime)

    if impute_method == 'backfill':
        temp = temp.resample('M').fillna("backfill")
        temp['value'] = temp['value'].pct_change(periods=1)
        temp = temp.dropna()
        temp.index = temp.index.map(round_datetime)

    if impute_method == 'linear_interpolate':
        temp=temp.resample('M').first()
        temp.index = temp.index.map(round_datetime)
        temp['value'] = temp['value'].interpolate('linear')
        temp['value'] = temp['value'].pct_change(periods=1)
        temp = temp['value'].dropna()
    return temp


def clean_data(fred, financial_data, chosen_indicators, impute_method):
    """
    Performs the Cleaning for all chosen macroeconomic indicators, combines it with the financial data
    :param fred: fred api connection object
    :param financial data: pandas dataframe, can be computed with collect_financial_data function
    :param chosen_indicators: dictionary, with keys indicating name of desired (according to fredapi database)
    macroeconomic series and values 'earliest_available' or '1_month_lag' indicating desired lag 
    :return: pandas dataframe of asssembled data
    """
    
    final = financial_data.copy()
    for indicator in chosen_indicators.keys():
        
        if chosen_indicators[indicator] == 'earliest_available':

            temp = clean_release_data(fred, indicator, impute_method)
            final = final.merge(temp.to_frame(indicator), left_index=True, right_index=True, how='left')
            if indicator == 'GDPC1':
                temp = clean_release_data(fred, indicator, impute_method, lag='date')
                final = final.merge(temp.to_frame('GDP_y'), left_index=True, right_index=True, how='left')
            
        if chosen_indicators[indicator] == '1_month_lag':
            temp = fred.get_series(indicator).resample('M').mean().shift(1).dropna()
            temp.index = temp.index.map(round_datetime)
            temp = temp.pct_change(periods=1)
            temp = temp.dropna()

            final = final.merge(temp.to_frame(indicator), left_index=True, right_index=True, how='left')

    return final

def compute_pca(X, perc_var_explained=0.9):
    """PCA Reduces a Numpy Matrix to first number of components explaining over perc_var_explained"""

    pca_train = X[:int(0.7 * X.shape[0])]

    pca_dims = PCA()
    pca_dims.fit(pca_train)
    cumsum = np.cumsum(pca_dims.explained_variance_ratio_)
    d = np.argmax(cumsum >= perc_var_explained) + 1 # d is the number of pca dimensions which preserves >pca_var*100% of original image variation
    print(f"{d} components are necessary to explain > {perc_var_explained*100}% of the image variation")
    pca = PCA(n_components=d)
    X_reduced = pca.fit_transform(X)
    X_recovered = pca.inverse_transform(X_reduced)

    return X_recovered

def create_windowed_data(X, y, window_size=12):
    """Creates windowed data and splits it into train validation and test sets"""
    
    X_data = []
    y_data = []
    for obs in range(window_size, X.shape[0]):
        X_data.append(X[obs-window_size:obs,:])       
    for obs in range(window_size, y.shape[0]):
        y_data.append(y[obs])

    X_data, y_data = np.array(X_data), np.array(y_data)

    n = y_data.shape[0]
    n_train, n_val = int(0.7*n), int(0.9*n)
    # Reshaping
    X_train = np.repeat(X_data[0:n_train], 100, axis=0)
    y_train = np.repeat(y_data[0:n_train], 100, axis=0)

    y_val = y_data[n_train:n_val]
    X_val = X_data[n_train:n_val]

    y_test = y_data[n_val:]
    X_test = X_data[n_val:]

    return X_data, y_data, X_train, y_train, X_val, y_val, X_test, y_test

def keras_RNN(input_shape, hidden_neurons, dropout, n_layers, activation):
    """
    Defines keras LSTM model
    :param input_shape: tuple of input shape
    :param hidden_neurons: int, number of neurons per layer
    :param dropout: dropout probability for dropout layers after each vertically stacked LSTM layer, None supported
    :param n_layers: int, number of total vertically stacked LSTM layers
    :param activation: keras activation function
    :return: assembled keras model
    """

    model = keras.models.Sequential()

    for n in range(0, n_layers):

        if n == 0:
            if n_layers == 1: 
                model.add(keras.layers.LSTM(hidden_neurons, batch_input_shape=input_shape))
            else: 
                model.add(keras.layers.LSTM(hidden_neurons, batch_input_shape=input_shape, return_sequences=True))
        elif n < n_layers-1: 
            model.add(keras.layers.LSTM(hidden_neurons, return_sequences=True))
        elif n == n_layers-1:
            model.add(keras.layers.LSTM(hidden_neurons))

        if dropout != None: 
            model.add(keras.layers.Dropout(dropout))

    model.add(keras.layers.Dense(1))
    model.add(keras.layers.Activation(activation))

    return model

def perform_training_validation(DATA_DIR, OUTPUT_DIR, gs_vars, gs_hyperparams, chosen_macro_indicators, fred, financial_data, flush=False):
    """
    Performs training, and validates, stores grid search results and visualizations for each grid iteration
    :param DATA_DIR: string, Data directory location
    :param OUTPUT_DIR: string, Visualisation directory location
    :param gs_vars: dictionary, keys are names of the variable group and values are names of the macroeconomic indicators
    :param gs_hyperparams: list of lists, contains lists of each hyperparameter search iteration
    :param fred: fred api connection object
    :param financial_data: pandas dataframe, can be computed with collect_financial_data function
    :param flush: boolean, whether to flush the existing grid search result pickle
    :return: pandas dataframe, contains grid search results
    """
    
    pkl_name = DATA_DIR+'gs_res.pkl'
    if os.path.isfile(pkl_name) and flush==False:
        with open(pkl_name, 'rb') as f:
            gs_res = pkl.load(f)
    else: 
        gs_res = []

    for cols in gs_vars.keys():
        for idx, (pca_opt, chosen_loss, fill, window_size, dropout, n_layers, n_neurons) in enumerate(gs_hyperparams): 

            print("=================================================")
            print("Presenting Results for: %s/%s Hyperparameter Combination" % (idx, len(gs_hyperparams)))
            print("Variables Chosen: %s" % cols)
            print("With %s" % pca_opt)
            print("Loss: %s" % chosen_loss)
            print("Filling Method: %s" % fill)
            print("Window Size: %s" % window_size)
            print("Dropout: %s" % dropout)
            print("N Layers: %s" % n_layers)
            print("N neurons: %s" % n_neurons)

            # Collect Fred API Macroeconomic variables
            series = clean_data(fred, financial_data, chosen_macro_indicators, fill).fillna(method='backfill')
            print("The data has %s observations" % series.shape[0])

            # Subset based on selected variables
            chosen_cols = gs_vars[cols]

            # Define X, y pairs
            X, y = np.array(series[chosen_cols]), np.array(series['GDP_y'])

            # Perform PCA?
            if pca_opt == 'PCA': X = compute_pca(X)

            # Define Training procedure
            X_data, y_data, X_train, y_train, X_val, y_val, X_test, y_test = create_windowed_data(X, y, window_size)
            num_features = X_train.shape[2]

            input_shape = (None, X_train.shape[1], X_train.shape[2])

            rnn = keras_RNN(input_shape, n_neurons, dropout, n_layers, keras.activations.linear)
            opt = keras.optimizers.Adam(learning_rate=1e-4)
            rnn.compile(optimizer = opt, loss = chosen_loss)

            batch = 1
            es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
            history = keras.callbacks.History()

            # Train!
            rnn.fit(X_train, y_train, validation_data=(X_val, y_val), epochs = 300, batch_size = int(batch), steps_per_epoch=1000, callbacks=[es, history]) 

            # Store test mse and validation mse
            validation_error = history.history['val_loss'][-1]
            GDP_test = rnn.predict(X_test)
            test_error = mean_squared_error(y_test, GDP_test)

            # Plot predictions
            GDP_preds = rnn.predict(X_data)

            f = plt.figure()
            plt.plot(series.iloc[window_size:].index, GDP_preds, label='Predicted')
            plt.plot(series.iloc[window_size:].index, y_data, label='True')
            plt.title('Validation Error: %0.6f Test Error: %0.6f' % (validation_error, test_error))
            plt.legend()
            plt.show()
            f.savefig(OUTPUT_DIR + "%s %s %s %s window %s dropout %s layers %s neurons %s.pdf" % (cols, pca_opt, chosen_loss, fill, window_size, dropout, n_layers, n_neurons), bbox_inches='tight')

            # Save Grid Search Iteration result
            curr_gs_res = [cols, pca_opt, chosen_loss, fill, window_size, validation_error, test_error, dropout, n_layers, n_neurons]
            gs_res.append(curr_gs_res)
            with open(pkl_name, 'wb') as f:
                pkl.dump(gs_res, f)

            print("=================================================")

    return pd.DataFrame(gs_res, columns=['variables','pca','loss','fill','window','validation_error','test_error', 'dropout','n_layers','n_neurons'])