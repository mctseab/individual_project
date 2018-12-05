import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from xgboost import XGBRegressor
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew #for some statistics
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
# from sklearn import cross_validation, linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


# from keras.models import Sequential
# from keras.layers import Dense

# import tensorflow as tf
# import itertools

import os
print(os.listdir("../input"))

# Load the data
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

train_df.head()

all_feature = ['penalty','l1_ratio','alpha','max_iter','random_state', 
                            'n_jobs', 'n_samples', 'n_features','n_classes', 
                            'n_clusters_per_class', 'n_informative', 'flip_y', 'scale']
nonnumeric_columns = ['penalty']

train_df = pd.get_dummies(train_df)
test_df = pd.get_dummies(test_df)
# train_df.info()

#n_jobs => -1 to be 16 #important
train_df['n_jobs'][train_df['n_jobs']== -1] = 16 #max
test_df['n_jobs'][test_df['n_jobs']== -1] = 16 #max

#Since test set has extra range of 'n_jobs' , 'n_samples' and 'n_features'
#Data expansion, Dummy samples
extra_X = pd.DataFrame(columns = list(test_df.columns))
extra_X.head()
extra_X.loc(0)[400] = test_df.loc(0)[1]
extra_X.loc(0)[401] = test_df.loc(0)[3]
extra_X.loc(0)[402] = test_df.loc(0)[5]
extra_X.loc(0)[403] = test_df.loc(0)[6]
extra_X.loc(0)[404] = test_df.loc(0)[8]
extra_X.loc(0)[405] = test_df.loc(0)[9]
extra_X.loc(0)[406] = test_df.loc(0)[11]
extra_X.loc(0)[407] = test_df.loc(0)[12]
extra_X.loc(0)[408] = test_df.loc(0)[13]
extra_X.loc(0)[409] = test_df.loc(0)[14]
extra_X.loc(0)[410] = test_df.loc(0)[16]
extra_X.loc(0)[411] = test_df.loc(0)[23]
extra_X.loc(0)[412] = test_df.loc(0)[25]
extra_X.loc(0)[413] = test_df.loc(0)[28]
extra_X.loc(0)[414] = test_df.loc(0)[30]
extra_X.loc(0)[415] = test_df.loc(0)[31]
extra_X.loc(0)[416] = test_df.loc(0)[45]
extra_X.loc(0)[417] = test_df.loc(0)[47]
extra_X.loc(0)[418] = test_df.loc(0)[51]
extra_X.loc(0)[419] = test_df.loc(0)[52]
extra_X.loc(0)[420] = test_df.loc(0)[56]
extra_X.loc(0)[421] = test_df.loc(0)[63]
extra_X.loc(0)[422] = test_df.loc(0)[64]
extra_X.loc(0)[423] = test_df.loc(0)[65]
extra_X.loc(0)[424] = test_df.loc(0)[68]
extra_X.loc(0)[425] = test_df.loc(0)[78]
extra_X.loc(0)[426] = test_df.loc(0)[84]
extra_X.loc(0)[427] = test_df.loc(0)[86]
extra_X.loc(0)[428] = test_df.loc(0)[91]
extra_X.loc(0)[429] = test_df.loc(0)[99]

extra_X_time = np.array([
                10.22,#1
                1.7135,#3
                9.218,#5
                2.507,#6
                14.9255,#8
                0.4925,#9
                13.442,#11
                1.01,#12
                32.6525,#13
                0.2585,#14
                0.6065,#16
                0.809,#23
                2.4185,#25
                2.1065,#28
                20.483,#30
                3.119,#31
                1.4705,#45
                0.5105,#47
                19.634,#51
                0.1055,#52
                1.007,#56
                7.0325,#63
                3.71,#64
                9.1325,#65
                8.219,#68
                3.9095,#78
                1.304,#84
                5.1185,#86
                1.2635,#91
                0.1175,#99
                ])

extra_X['time'] = pd.Series(extra_X_time, index=extra_X.index)
# extra_X.head()
train_df = train_df.append(extra_X)#<<<<<TODO
# train_df.head()
# train_df.loc(0)[410]

y = train_df['time']
train_df.drop('time',axis = 1, inplace = True)
combined_X = train_df.append(test_df)
combined_X.drop('id',axis = 1, inplace = True)
# combined_X.info()
combined_X.head()


#drop useless features
combined_X.drop('random_state',axis = 1, inplace = True)
combined_X.drop('flip_y',axis = 1, inplace = True)
combined_X.drop('scale',axis = 1, inplace = True)
combined_X.drop('alpha',axis = 1, inplace = True)# 0.01, 0.001, 0.0001
combined_X.drop('n_informative',axis = 1, inplace = True) #
# combined_X.drop('n_clusters_per_class',axis = 1, inplace = True)#<<<ok
combined_X.drop('l1_ratio',axis = 1, inplace = True)###
# combined_X.drop('n_classes',axis = 1, inplace = True)
# combined_X.drop('n_jobs',axis = 1, inplace = True)
# combined_X.drop('penalty_none',axis = 1, inplace = True)
# combined_X.drop('penalty_l2',axis = 1, inplace = True)
# combined_X.drop('penalty_l1',axis = 1, inplace = True)
# combined_X.drop('penalty_elasticnet',axis = 1, inplace = True)
# combined_X.drop('n_samples',axis = 1, inplace = True)
# combined_X.drop('n_features',axis = 1, inplace = True)
# combined_X.drop('max_iter',axis = 1, inplace = True)


########################
#check missing data
combined_X.isnull().sum()


########################
#check correlation
# corrmat = train_df.corr()
# # corrmat = combined_X.append(y).corr()
# plt.subplots(figsize=(12,9))
# sns.heatmap(corrmat, vmax=0.9, square=True)

# ########################
# #check skew feature
# numeric_feats = combined_X.dtypes[combined_X.dtypes != "object"].index
# # Check the skew of all numerical features
# skewed_feats = combined_X[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
# print("\nSkew in numerical features: \n")
# skewness = pd.DataFrame({'Skew' :skewed_feats})
# skewness.head(10)

# ########################
# #do Box Cox transform
# skewness = skewness[abs(skewness) > 0.75]
# print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
# from scipy.special import boxcox1p
# skewed_features = skewness.index
# lam = 0.15
# for feat in skewed_features:
#     combined_X[feat] = boxcox1p(combined_X[feat], lam)
# print("finished.")

########################

# Scaling, MinMaxScaler #TODO
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(combined_X)
normalized_X = scaler.transform(combined_X)
# X_val_scld = scaler.transform(X_val)
# X_test_scld = scaler.transform(X_test)
normalized_X = pd.DataFrame(normalized_X, columns = list(combined_X.columns))
normalized_X.head()


mean_y = y.mean()
variance_y = y.var()
std_y = y.std()
# mean_y
# variance_y
normalized_y = (y - mean_y)/variance_y
# normalized_y = (train_df['time'] - mean_y)/std_y
# plt.plot(normalized_y)
# normalized_y.shape[0]



sns.distplot( y, fit=norm)
# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(y)
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('time distribution')
#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(y, plot=plt)
plt.show()

#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
# y = np.log1p(y)
y = np.log(y)
#Check the new distribution 
sns.distplot( y, fit=norm);
# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(y)
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('time distribution')
#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(y, plot=plt)
plt.show()

normalized_y = (y - mu)/sigma


#PCA
# pca = PCA().fit(normalized_X)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# # plt.xlim(0,13,1)
# plt.xlabel('Number of components')
# plt.ylabel('Cumulative explained variance')


feature_columns_to_use_1 = ['penalty','l1_ratio','alpha','max_iter','random_state', 
                            'n_jobs', 'n_samples', 'n_features','n_classes', 
                            'n_clusters_per_class', 'n_informative', 'flip_y', 'scale']
feature_columns_to_use_2 = ['penalty','max_iter','random_state', 
                            'n_jobs', 'n_samples', 'n_features', 'flip_y']
feature_columns_to_use_3 = ['penalty','max_iter', 'n_jobs', 'n_samples', 'n_features']
feature_columns_to_use_4 = ['penalty','max_iter', 'n_samples', 'n_features']


# normalized_X[feature_columns_to_use].head()

selected_X = normalized_X
# selected_X = pc_X #used PCA
X = selected_X[0:train_df.shape[0]].as_matrix()#for normal
# X = selected_X[0:train_df.shape[0]]#for PCA
# y = train_df['time']
y = normalized_y
# y = y

test_X_submit = selected_X[train_df.shape[0]::].as_matrix() #normal
predictions_submit_sum = 0
predictions_submit = 0
real_prediction_submit = 0

# XGBoost parameters
xgboost_params = {
    'max_depth':3, #4 or 5 [3,4,5,6,7,8,9,10]
    'learning_rate':0.1,  #[0.01,0.03,0.1,0.3,1]
    'n_estimators':300, #300 [10,30,100,300,1000]
    'silent':True, 
    'objective':'reg:linear', 
    'booster':'gbtree', #gbtree, gblinear, dart
    'n_jobs':10,
    'nthread':None, 
    'gamma':0, #[0.01, 0.03, 0.1, 0.3, 1]#dont care
    'min_child_weight':5, #[1, 3, 10, 20, 30, 100, 300]
    'max_delta_step':0, # [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300] #dont care
    'subsample':1, #[0.01, 0.03, 0.1, 0.3, 1]
    'colsample_bytree':1, #[0.01, 0.03, 0.1, 0.3, 1]
    'colsample_bylevel':0.6, #[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    'reg_alpha':0.1, #[0.01, 0.03, 0.1, 0.3, 1]
    'reg_lambda':0.2, #[0.1, 0.3, 1, 3, 10, 30, 100]
    'scale_pos_weight':0.7, #[0.1,0.3,0.5,0.7,1, 3]
    'base_score':1, #[0.01,0.03,0.1, 0.3, 1, 3]
    'random_state':0, 
    'seed':None, 
    'missing':None
}

# hyper_params = [500,1000, 1500, 2000, 5000, 10000]#10000
# hyper_params = [3, 5, 10, 30, 100]#100
# hyper_params = [2200,3000,4000]#n_estimators, higher better
hyper_params = [1,3,5,7, 10, 20, 30]


plt.figure()
plt.title('CV vs seeds')
is_test_param = False
is_train_all = False
mse_for_different_seed = []
seed_size = 5
for seed in range(seed_size):
    print("Seed={}".format(seed))
    all_mse_for_params = []
    for param in hyper_params:
        all_mse_for_params.append([])
        
    K = 10#10
    if is_train_all == False:
        kf = KFold(n_splits = K, random_state = seed+50, shuffle = True)#30,90,150
        kf_mse = []
        k_index = 0
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
                
            k_index += 1
            param_index = 0
            
            ##temp
            # history =  model.fit(np.array(X_train),np.array(y_train), epochs=100, \
            #              batch_size=15, validation_data=(np.array(X_test),np.array(y_test)),\
            #              verbose=1, shuffle=False,callbacks=None)
            # scores=model.evaluate(np.array(X_train),np.array(y_train))
            # print("\n%s %.2f" % (model.metrics_names[1],scores[1]))
    
            ##temp
            
            if is_test_param:
                for param in hyper_params:
                    # xgb_regr = XGBRegressor(learning_rate=param)#max_depth=7
                    # xgb_regr = XGBRegressor(n_estimators=1000, learning_rate=0.01, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7)#max_depth=7
                    xgb_regr = XGBRegressor(
                                        # objective=xgboost_params['objective'],
                                        # booster=xgboost_params['booster'],
                                        max_depth=xgboost_params['max_depth'], 
                                        learning_rate=xgboost_params['learning_rate'], 
                                        n_estimators=param,#xgboost_params['n_estimators'], 
                                        gamma=xgboost_params['gamma'], 
                                        min_child_weight=xgboost_params['min_child_weight'], 
                                        max_delta_step=xgboost_params['max_delta_step'], 
                                        subsample=xgboost_params['subsample'], 
                                        colsample_bytree=xgboost_params['colsample_bytree'], 
                                        colsample_bylevel=xgboost_params['colsample_bylevel'], 
                                        reg_alpha=xgboost_params['reg_alpha'], 
                                        reg_lambda=xgboost_params['reg_lambda'], 
                                        scale_pos_weight=xgboost_params['scale_pos_weight'], 
                                        base_score=xgboost_params['base_score']
                                        )#after kfold
                    # xgb_regr.fit(X_train,y_train)
                    
                    # lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
                    # ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
                    # KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
                    # GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                    #                   max_depth=4, max_features='sqrt',
                    #                   min_samples_leaf=15, min_samples_split=10, 
                    #                   loss='huber', random_state =5)
                    # model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                    #                               learning_rate=0.05, n_estimators=720,
                    #                               max_bin = 55, bagging_fraction = 0.8,
                    #                               bagging_freq = 5, feature_fraction = 0.2319,
                    #                               feature_fraction_seed=9, bagging_seed=9,
                    #                               min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
                    model_lgb = lgb.LGBMRegressor(
                                    boosting_type='gbdt', #gbdt, dart,goss, rf
                                    max_depth=4,#-1
                                    subsample_for_bin=200000,#200000,
                                    reg_alpha=0.0003, 
                                    reg_lambda=100,
                                     min_split_gain=0.0001,#0.0, 
                                    #  min_child_weight=0.001, 
                                    #  min_child_samples=20,#20, 
                                    #  subsample=1.0, 
                                    #  subsample_freq=0, 
                                    #  colsample_bytree=1.0,
                                    objective='regression',
                                    num_leaves=5,  #23.35234
                                    learning_rate=0.3,#0.3, 
                                    n_estimators=3000,#10000
                                    max_bin = 10, 
                                    bagging_fraction = 0.8,
                                    bagging_freq = 5, 
                                    feature_fraction = 0.5,
                                    feature_fraction_seed=9, 
                                    bagging_seed=9,
                                    min_data_in_leaf =6, 
                                    min_sum_hessian_in_leaf = 11)
                    
                    # model_xgb = xgb.XGBRegressor(colsample_bytree=1, gamma=0,  #
                    #                          colsample_bylevel=0.6, #TODO for overfit
                    #                          base_score=0.54, #[0.5, 0.53,0.54,0.55,0.56,0.57 ]
                    #                          scale_pos_weight=1.39, #1.02 #[1, 1.02, 1.03, 1.04, 1.05 ]
                    #                         #  max_delta_step=param, #dont care
                    #                          learning_rate=0.045, 
                    #                          max_depth=6, #max_depth >3 TODO [3,5,6,7]
                    #                          min_child_weight=10.5, #min_child_weight TODO
                    #                          n_estimators=2000, # n_estimators >4000, [2200, 4000, 8000,10000], overfit
                    #                          reg_alpha=0.4640, #overfit?
                    #                          reg_lambda=param, #reg_lambda=7 TODO for overfit
                    #                          subsample=0.53, silent=1, #subsample TODO 0.52 for overfit
                    #                          random_state =7, nthread = -1)
                                             
                    # GBoost = GradientBoostingRegressor(
                    #             n_estimators=3000, #3000
                    #             learning_rate=0.1, #[0.05, 0.06, 0.07,0.08,0.1,0.11]
                    #             max_depth=5, #5 #TODO [4,5,6,7]
                    #             max_features=2, #2 or 'sqrt' ['auto', 'sqrt', 'log2'], max_features < n_features leads to a reduction of variance and an increase in bias.
                    #             min_samples_leaf=0.08,#15,0.08 TODO, fraction / int, #[0.01,0.07,0.08,0.09,0.1,0.15 ]
                    #             min_samples_split=0.08, #[0.01,0.1,0.2 ]
                    #             loss='huber', #huber ['ls', 'lad', 'huber', 'quantile'] 
                    #             subsample=1, #TODO 0.98, [0.96,0.97,0.98,1] # < 1.0 leads to a reduction of variance and an increase in bias.
                    #             criterion='friedman_mse', # ['friedman_mse', 'mse', 'mae']
                    #             # min_weight_fraction_leaf=0, #0 #[0, 0.001, 0.01]
                    #             # min_impurity_decrease=0,#0
                    #             # init=None, 
                    #             alpha=0.99, #0.99 #[0.89,0.9, 0.99,0.995]
                    #             # max_leaf_nodes=None, 
                    #             # warm_start=param, #[True, False]
                    #             # presort=param, #'auto' [True, False, 'auto']
                    #             validation_fraction=0.07, #0.07, [0.01,0.05, 0.1,0.11]
                    #             n_iter_no_change=100, #None, [50,100,300]
                    #             tol=0.00001, #0.0001, #[0.00001, 0.0001, 0.001]
                    #             random_state = 5)
                                
                    lasso = make_pipeline(RobustScaler(), Lasso(
                                alpha =0.0005, #0.0005
                                random_state=1)) #overfit *** 
                    ENet = make_pipeline(RobustScaler(), ElasticNet(
                                alpha=0.0005, #0.0005
                                l1_ratio=0.9, #0.9, 0
                                random_state=3)) #overfit ***
                    KRR = KernelRidge(
                                alpha=0.6, 
                                kernel='polynomial', 
                                degree=2, 
                                coef0=2.5,
                                ) #overfit *****
    
                    model = model_lgb
                    model.fit(X_train,y_train)
                    predictions = model.predict(X_test)
                    
                    
                    # mse = ((predictions*variance_y) + mean_y) - ((y_test*variance_y) + mean_y)
                    # mse = np.mean(mse ** 2)
                    # mse = np.mean((predictions-y_test) ** 2)
                    # mse = np.mean((np.expm1(predictions*sigma+mu) - np.expm1(y_test*sigma+mu)) ** 2)
                    
                    # mse = np.mean((np.expm1(predictions*sigma+mu) - np.expm1(y_test*sigma+mu)) ** 2)#<<<<<
                    mse = np.mean((np.exp(predictions*sigma+mu) - np.exp(y_test*sigma+mu)) ** 2)
                    # print("MSE: %.2f" % mse)
                    # kf_mse.append(mse)
                    all_mse_for_params[param_index].append(mse)
                    # print(predictions) #test
                    param_index += 1
            else: #run for submission
                xgb_regr = XGBRegressor(
                                        # objective=xgboost_params['objective'],
                                        booster=xgboost_params['booster'],
                                        max_depth=xgboost_params['max_depth'], 
                                        learning_rate=xgboost_params['learning_rate'], 
                                        n_estimators=xgboost_params['n_estimators'], 
                                        gamma=xgboost_params['gamma'], 
                                        min_child_weight=xgboost_params['min_child_weight'], 
                                        max_delta_step=xgboost_params['max_delta_step'], 
                                        subsample=xgboost_params['subsample'], 
                                        colsample_bytree=xgboost_params['colsample_bytree'], 
                                        colsample_bylevel=xgboost_params['colsample_bylevel'], 
                                        reg_alpha=xgboost_params['reg_alpha'], 
                                        reg_lambda=xgboost_params['reg_lambda'], 
                                        scale_pos_weight=xgboost_params['scale_pos_weight'], 
                                        base_score=xgboost_params['base_score'],
                                        # max_features='sqrt',
                                        # loss='huber',
                                        random_state=seed,
                                        # sedd=seed,
                                        )#after kfold
                # xgb_regr.fit(X_train,y_train)
                # predictions = xgb_regr.predict(X_test)
                
                lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1)) #overfit *** 
                ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)) #overfit ***
                KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5) #overfit *****
                GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, #19.94329 #overfit **
                                  max_depth=4, max_features='sqrt',
                                  min_samples_leaf=15, min_samples_split=10, 
                                  loss='huber', random_state =5)
                model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,  #23.77381 #overfit **
                                             learning_rate=0.05, max_depth=3, 
                                             min_child_weight=1.7817, n_estimators=2200,
                                             reg_alpha=0.4640, reg_lambda=0.8571,
                                             subsample=0.5213, silent=1,
                                             random_state =7, nthread = -1)
                model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,  #23.35234 #overfit **
                                              learning_rate=0.05, n_estimators=720,
                                              max_bin = 55, bagging_fraction = 0.8,
                                              bagging_freq = 5, feature_fraction = 0.2319,
                                              feature_fraction_seed=9, bagging_seed=9,
                                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
                                    
                #tuned##################          
                # model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=3,  #23.35234
                #                               learning_rate=0.3, n_estimators=3000,#10000
                #                               max_bin = 10, bagging_fraction = 0.8,
                #                               bagging_freq = 5, feature_fraction = 0.5,
                #                               feature_fraction_seed=9, bagging_seed=9,
                #                               min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
                model_lgb = lgb.LGBMRegressor(
                                    boosting_type='gbdt', #gbdt, dart,goss, rf
                                    max_depth=4,#-1
                                    subsample_for_bin=200000,#200000,
                                    reg_alpha=0.0003, 
                                    reg_lambda=100,
                                     min_split_gain=0.0001,#0.0, 
                                    #  min_child_weight=0.001, 
                                    #  min_child_samples=20,#20, 
                                    #  subsample=1.0, 
                                    #  subsample_freq=0, 
                                    #  colsample_bytree=1.0,
                                    objective='regression',
                                    num_leaves=5,  #23.35234
                                    learning_rate=0.3,#0.3, 
                                    n_estimators=3000,#10000
                                    max_bin = 10, 
                                    bagging_fraction = 0.8,
                                    bagging_freq = 5, 
                                    feature_fraction = 0.5,
                                    feature_fraction_seed=9, 
                                    bagging_seed=9,
                                    min_data_in_leaf =6, 
                                    min_sum_hessian_in_leaf = 11)
                model_xgb = xgb.XGBRegressor(colsample_bytree=1, gamma=0,  #23.77381 #overfit **
                                             colsample_bylevel=0.6, #TODO for overfit
                                             base_score=0.54, #[0.5, 0.53,0.54,0.55,0.56,0.57 ]
                                             scale_pos_weight=1.39, #1.02 #[1, 1.02, 1.03, 1.04, 1.05 ]
                                            #  max_delta_step=param, #dont care
                                             learning_rate=0.045, 
                                             max_depth=6, #max_depth >3 TODO [3,5,6,7]
                                             min_child_weight=10.5, #min_child_weight TODO
                                             n_estimators=2000, # n_estimators >4000, [2200, 4000, 8000,10000]
                                             reg_alpha=0.4640, reg_lambda=7, #reg_lambda=7 TODO for overfit
                                             subsample=0.53, silent=1, #subsample TODO 0.52 for overfit
                                             random_state =7, nthread = -1)
                GBoost = GradientBoostingRegressor(
                                n_estimators=3000, #3000
                                learning_rate=0.1, #[0.05, 0.06, 0.07,0.08,0.1,0.11]
                                max_depth=5, #5 #TODO [4,5,6,7]
                                max_features=2, #2 or 'sqrt' ['auto', 'sqrt', 'log2'], max_features < n_features leads to a reduction of variance and an increase in bias.
                                min_samples_leaf=0.08,#15,0.08 TODO, fraction / int, #[0.01,0.07,0.08,0.09,0.1,0.15 ]
                                min_samples_split=0.08, #[0.01,0.1,0.2 ]
                                loss='huber', #huber ['ls', 'lad', 'huber', 'quantile'] 
                                subsample=1, #TODO 0.98, [0.96,0.97,0.98,1] # < 1.0 leads to a reduction of variance and an increase in bias.
                                criterion='friedman_mse', # ['friedman_mse', 'mse', 'mae']
                                # min_weight_fraction_leaf=0, #0 #[0, 0.001, 0.01]
                                # min_impurity_decrease=0,#0
                                # init=None, 
                                alpha=0.99, #0.99 #[0.89,0.9, 0.99,0.995]
                                # max_leaf_nodes=None, 
                                # warm_start=param, #[True, False]
                                # presort=param, #'auto' [True, False, 'auto']
                                validation_fraction=0.07, #0.07, [0.01,0.05, 0.1,0.11]
                                n_iter_no_change=100, #None, [50,100,300]
                                tol=0.00001, #0.0001, #[0.00001, 0.0001, 0.001]
                                random_state = 5)
                
                model = model_lgb
                model.fit(X_train,y_train)
                predictions = model.predict(X_test)
                
                
                # mse = np.mean((np.expm1((predictions*sigma)+mu) - np.expm1((y_test*sigma)+mu)) ** 2)
                mse = np.mean((np.exp((predictions*sigma)+mu) - np.exp((y_test*sigma)+mu)) ** 2)
                all_mse_for_params[0].append(mse)
                
                #average prediction
                predictions_submit = model.predict(test_X_submit)
                predictions_submit = np.exp((predictions_submit*sigma) +mu)
                for n, i in enumerate(predictions_submit):
                    if i < 0:
                        predictions_submit[n] = 0
                predictions_submit_sum += predictions_submit
                
                if seed == 3:
                    real_prediction_submit += predictions_submit
    else:
        model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=3,  #23.35234
                                              learning_rate=0.3, n_estimators=3000,#10000
                                              max_bin = 10, bagging_fraction = 0.8,
                                              bagging_freq = 5, feature_fraction = 0.5,
                                              feature_fraction_seed=9, bagging_seed=9,
                                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11,
                                              random_state=seed)
        model = model_lgb
        model.fit(X,y)
        predictions = model.predict(X)
        
        
        # mse = np.mean((np.expm1((predictions*sigma)+mu) - np.expm1((y*sigma)+mu)) ** 2)
        mse = np.mean((np.exp((predictions*sigma)+mu) - np.exp((y*sigma)+mu)) ** 2)
        all_mse_for_params[0].append(mse)
        
        #average prediction
        predictions_submit = model.predict(test_X_submit)
        # predictions_submit = np.expm1((predictions_submit*sigma) +mu)
        predictions_submit = np.exp((predictions_submit*sigma) +mu)
        for n, i in enumerate(predictions_submit):
            if i < 0:
                predictions_submit[n] = 0
        predictions_submit_sum += predictions_submit                                   
    
    
    average_k_mse = []
    if is_test_param:
        for k_mse in all_mse_for_params:
            average_k_mse.append(np.mean(k_mse))        
        
        plt.plot(hyper_params, average_k_mse,'o--', label='seed={}'.format(seed))
        plt.xlabel('hyper_params')
        plt.ylabel('mse')
        plt.show()
    else:
        mse_for_different_seed.append(np.mean(all_mse_for_params[0])) 
        if seed == 3:
            real_prediction_submit /= K

if is_test_param == False:
    plt.plot(mse_for_different_seed)
    plt.xlabel('seed')
    plt.ylabel('mse')
    plt.show()
# plt.legend(loc='upper right') 
# plt.show()

#################################
########## Submission ###########
if is_train_all == False:
    predictions_submit = predictions_submit_sum/(seed_size * K)
else:
    predictions_submit = predictions_submit_sum/(seed_size)


print(predictions_submit)
plt.plot(predictions_submit)
submission = pd.DataFrame({ 'id': test_df['id'], 'time': predictions_submit })
submission.to_csv("submission.csv", index=False)
