import joblib
import pandas as pd
import numpy as np

RANDOM_STATE = 42

df = pd.read_csv('data/real_estate_thesis_processed.csv', low_memory=False, index_col=0)

general_ranking = pd.read_csv('data/feature_ranking.csv', index_col=0)

selected_cols = general_ranking[general_ranking['keep']]

if 'outlier' in selected_cols.index:
    selected_cols = selected_cols.index.drop('outlier')

selected_cols = selected_cols.index.to_list()

df = df[df['price'].notna()] #.reset_index(drop=True)




######## Split

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.tree import plot_tree
from sklearn import metrics
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error
from sklearn import tree
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


if 'price_per_m2' in df.columns:
    df.drop(columns=['price_per_m2'], inplace=True)


target_variable = "price"

feature_names = df.columns.drop([target_variable])

vec_cols = [col for col in feature_names if col.startswith('vector')]
poi_cols = [col for col in selected_cols if col.startswith('dist_') or col.startswith('count_') or col.startswith('play_areas_') or col.startswith('park_areas_') ]
panorama_cols = [col for col in selected_cols if col.startswith('pano_')]
base_cols = [col for col in selected_cols if col not in poi_cols and col not in panorama_cols]

#df_sample = df.copy()

#cond_filter = df['price']<2000000
df_sample=df[~df['outlier']] #.reset_index(drop=True)

df_sample.drop(columns=['outlier'], inplace=True)


price_bins = [0, 250000, 500000, 1000000, 1500000, 2000000, 3000000, float("inf")]
price_labels = ['0-250k', '250-500k', '500k-1M', '1M-1.5M', '1.5M-2M', '2M-3M', '3M+']

df_sample['price_bin'] = pd.cut(df_sample[target_variable], bins=price_bins, labels=price_labels)

X_train, X_test, Y_train, Y_test = train_test_split(
    df_sample[selected_cols], df_sample[target_variable],
    stratify=df_sample['price_bin'],
    random_state=RANDOM_STATE,
    test_size=0.20
)


Y_train_ln = np.log(Y_train)
Y_test_ln = np.log(Y_test)




#### Comparison of methods


N_ITER = 500
SCORING='neg_mean_absolute_error' # neg_mean_absolute_error, neg_root_mean_squared_error, neg_mean_absolute_percentage_error
CV = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, RobustScaler

log_transform_cols = ['area_m2', 'floor_number', 'building_age',
                       'building_floors_num', 'distance_to_center_km', 'room_size_m2', 'floor_ratio' ]

scale_cols = ['latitude', 'longitude',  ]



def get_pipeline(model):
    pipeline = Pipeline([
        #('preprocessor', preprocessor),
        ('model', model)
    ])
    return pipeline


from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error


def get_metrics(Y_test, Y_predict, which=None):
    rmse = root_mean_squared_error(Y_test, Y_predict)
    r2 = r2_score(Y_test, Y_predict)
    mape = mean_absolute_percentage_error(Y_test, Y_predict)

    #if which:
    #    print(f'{which} metrics')
    metrics_df = pd.DataFrame({
                            'rmse':[rmse],
                            'r2':[r2],
                            'mape':[mape],
                           })
    return metrics_df

results = dict()



import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

def compute_errors(X_train, Y_train, Y_predict_train, X_test, Y_test, Y_predict_test, price_bins=None):
    """
    Compute RMSE and MAPE per district and per price range for both training and test sets.
    Returns merged results in a DataFrame.
    """

    # Make copies to avoid modifying original DataFrames
    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()

    # Step 1: Reconstruct the 'district' column
    #district_cols = [col for col in X_test_copy.columns if col.startswith('district_')]
    #X_test_copy['district'] = X_test_copy[district_cols].idxmax(axis=1).str.replace('district_', '')
    #X_train_copy['district'] = X_train_copy[district_cols].idxmax(axis=1).str.replace('district_', '')

    # Step 2: Create DataFrames for train and test
    df_test = pd.DataFrame({'district': None, 'y_true': Y_test, 'y_pred': Y_predict_test})
    df_train = pd.DataFrame({'district': None, 'y_train': Y_train, 'y_pred_train': Y_predict_train})

    # Step 3: Compute errors per district
    def compute_group_errors(df, true_col, pred_col, groupby='district'):
        results = []
        for group_name, group in df.groupby(groupby):
            rmse = np.sqrt(mean_squared_error(group[true_col], group[pred_col]))
            mape = mean_absolute_percentage_error(group[true_col], group[pred_col])
            results.append({groupby: group_name, 'n': len(group), 'RMSE': int(rmse), 'MAPE': round(mape, 4)})
        return pd.DataFrame(results).sort_values(by='MAPE', ascending=True)

    #test_df = compute_group_errors(df_test, 'y_true', 'y_pred')
    #train_df = compute_group_errors(df_train, 'y_train', 'y_pred_train')

    # Merge test and train results per district
    #district_errors = pd.merge(test_df, train_df, on='district', how='left', suffixes=('_test', '_train'))

    # Step 4: Compute errors per price range if bins are provided
    if price_bins:
        df_test['price_range'] = pd.cut(df_test['y_true'], bins=price_bins)
        df_train['price_range'] = pd.cut(df_train['y_train'], bins=price_bins)

        test_price_errors = compute_group_errors(df_test, 'y_true', 'y_pred', groupby='price_range')
        train_price_errors = compute_group_errors(df_train, 'y_train', 'y_pred_train', groupby='price_range')

        price_errors = pd.merge(test_price_errors, train_price_errors, on='price_range', how='left', suffixes=('_test', '_train'))
    else:
        price_errors = None


    return price_errors







#####  testing with subsets of data


import xgboost as xgb

best_params = {'max_depth': 10, 'learning_rate': 0.0434191761314323, 'n_estimators': 894, 'min_child_weight': 2,
               'subsample': 0.6793586180035424, 'colsample_bytree': 0.7027061765951619, 'reg_alpha': 2.582618651372783,
               'reg_lambda': 6.94850833232177}



feature_sets = {
    "base": base_cols,
    #"poi": poi_cols,
    #"panorama": panorama_cols,
    "base_poi": base_cols + poi_cols,
    "base_panorama": base_cols + panorama_cols,
    "all": base_cols + poi_cols + panorama_cols
    # vectors images
}

results = {}
for name, cols in feature_sets.items():
    X_train_subset = X_train[cols]
    X_test_subset = X_test[cols]

    model = xgb.XGBRegressor(tree_method="hist", objective='reg:quantileerror', quantile_alpha=0.5, **best_params)
    model.fit(X_train_subset, Y_train_ln)
    joblib.dump(model, f'models/xgb_qo_{name}.pkl')


    Y_predict_train = np.exp(model.predict(X_train_subset))
    Y_predict_test = np.exp(model.predict(X_test_subset))

    r_train = get_metrics(Y_train, Y_predict_train, 'Train')
    r_test = get_metrics(Y_test, Y_predict_test, 'Test')

    r_train.index = ['Train']
    r_test.index = ['Test']

    score = r_test['rmse'].values[0]
    print(f"{name}: {score}")



    results[name] = score







from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

prefix = "model__"

param_dist = {
    f'{prefix}max_depth': randint(3, 200),
    f'{prefix}min_samples_split': [5, 10, 20, 50, 100],
    f'{prefix}min_samples_leaf': [5, 10, 20, 50],
    f'{prefix}max_leaf_nodes': [10, 20, 30, 50, 100],
    f'{prefix}min_impurity_decrease': uniform(0, 0.1),
    f'{prefix}ccp_alpha': uniform(0, 0.05)
}



param_dist = {
    f"{prefix}max_depth": [3, 5, 10, 20, None],
    f"{prefix}min_samples_split": [2, 5, 10],
    f"{prefix}min_samples_leaf": [1, 2, 4, 10],
    f"{prefix}max_features": ["sqrt", "log2", None]
}



dtr = RandomizedSearchCV(
    estimator=get_pipeline(DecisionTreeRegressor()),
    param_distributions=param_dist,
    n_iter=N_ITER,
    scoring=SCORING, # neg_root_mean_squared_error, neg_mean_absolute_percentage_error
    cv=CV,
    n_jobs=-1,
    verbose=1,
    random_state=RANDOM_STATE,
    error_score='raise'
)


dtr.fit(X_train, Y_train_ln)

import joblib

joblib.dump(dtr, 'models/decision_tree_model.pkl')


best_params = dtr.best_params_
print("Best Hyperparameters:", best_params)
print(f"Best score: {dtr.best_score_:.4f}")




dtr = joblib.load('models/decision_tree_model.pkl')

Y_predict_train = np.exp(dtr.predict(X_train))
Y_predict_test = np.exp(dtr.predict(X_test))

get_metrics(Y_train, Y_predict_train, 'Train')
results["DecisionTreeRegressor"] = get_metrics(Y_test, Y_predict_test, 'Test')





import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

prefix = "model__"

param_dist = {
    f'{prefix}max_depth': randint(3, 30),
    f'{prefix}learning_rate': uniform(0.01, 0.2),
    f'{prefix}n_estimators': randint(200, 500),
    f'{prefix}min_child_weight': randint(1, 10),

    f'{prefix}subsample': uniform(loc=0.5, scale=0.5),
    f'{prefix}colsample_bytree': uniform(loc=0.5, scale=0.5),

    f'{prefix}gamma': uniform(0, 0.5),
    f'{prefix}reg_lambda': uniform(1, 20),  # L2 regularization
    f'{prefix}reg_alpha': uniform(0, 10),   # L1 regularization
    f"{prefix}objective": ["reg:tweedie", "reg:pseudohubererror", "reg:squarederror"],
    f"{prefix}tree_method": ["approx", "hist"]
}



xgb_1 = RandomizedSearchCV(
    estimator=get_pipeline(xgb.XGBRegressor()),
    param_distributions=param_dist,
    n_iter=N_ITER,
    scoring=SCORING,
    cv=CV,
    n_jobs=-1,
    verbose=1,
    random_state=RANDOM_STATE,
    return_train_score=True
)

xgb_1.fit(X_train, Y_train_ln)

import joblib
joblib.dump(xgb_1, 'models/xgb_1_randcv.pkl')


print("Best Hyperparameters:",  xgb_1.best_params_)
print(f"Best score: {xgb_1.best_score_:.4f}")




import joblib

xgb_1 = joblib.load('models/xgb_1_randcv.pkl')

Y_predict_train = np.exp(xgb_1.predict(X_train))

Y_predict_test = np.exp(xgb_1.predict(X_test))


test_metrics = get_metrics(Y_test, Y_predict_test, 'Test')

results["XGBRegressor"] = test_metrics




price_bins = [250000, 500000, 750000, 1000000, 1500000, 2000000, np.inf]
price_errors = compute_errors(X_train, Y_train, Y_predict_train, X_test, Y_test, Y_predict_test, price_bins)

price_errors.sort_values(by='price_range')



best_xgb = xgb_1.best_estimator_.named_steps['model']
feature_importance = best_xgb.feature_importances_
feature_names = X_train.columns

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False).reset_index(drop=True)


importance_df.head(20)





import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint


prefix = "model__"


param_dist = {
    f"{prefix}n_estimators": randint(100, 600),
    f"{prefix}learning_rate": uniform(0.01, 0.3),
    f"{prefix}max_depth": randint(3, 15),
    f"{prefix}min_child_weight": randint(1, 10),
    f"{prefix}subsample": uniform(0.5, 0.5),
    f"{prefix}colsample_bytree": uniform(0.5, 0.5),
    f"{prefix}gamma": uniform(0, 5),
    f"{prefix}reg_lambda": uniform(0, 20),
    f"{prefix}reg_alpha": uniform(0, 10),
}



xgbq_model = xgb.XGBRegressor(tree_method="hist", objective='reg:quantileerror', quantile_alpha=0.5)


xbg_q = RandomizedSearchCV(
    estimator=get_pipeline(xgbq_model),
    param_distributions=param_dist,
    n_iter=N_ITER,
    scoring=SCORING,
    cv=CV,
    n_jobs=-1,
    verbose=1,
    random_state=RANDOM_STATE,
    return_train_score=True
)

xbg_q.fit(X_train, Y_train_ln)

import joblib
joblib.dump(xbg_q, 'models/xbg_q_randcv.pkl')

xbg_q.best_params_




xbg_q = joblib.load('models/xbg_q_randcv.pkl')

Y_predict_train = np.exp(xbg_q.predict(X_train))
Y_predict_test = np.exp(xbg_q.predict(X_test))

get_metrics(Y_train, Y_predict_train, 'Train')
results["XGBRegressorQuantile"] = get_metrics(Y_test, Y_predict_test, 'Test')


best_xgb = xbg_q.best_estimator_.named_steps['model']
feature_importance = best_xgb.feature_importances_
feature_names = X_train.columns

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False).reset_index(drop=True)


importance_df.head(20)





import optuna
import xgboost as xgb
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import numpy as np

def objective(trial):
    # Suggest hyperparameters
    # n_estimators = trial.suggest_int("n_estimators", 100, 500)
    # learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    # max_depth = trial.suggest_int("max_depth", 3, 15)
    # min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
    # subsample = trial.suggest_float("subsample", 0.5, 1.0)
    # colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    # gamma = trial.suggest_float("gamma", 0, 5)
    # reg_lambda = trial.suggest_float("reg_lambda", 0, 20)
    # reg_alpha = trial.suggest_float("reg_alpha", 0, 10)

    param = {
    'max_depth': trial.suggest_int('max_depth', 3, 10),
    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),  # L1 regularization
    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),  # L2 regularization
    'random_state': 42
    }

    # Define XGBRegressor
    model = make_pipeline(
        get_pipeline(xgb.XGBRegressor(
            tree_method="hist",
            objective="reg:quantileerror",
            quantile_alpha=0.5,
            **param
        ))
    )

    # Perform cross-validation
    scores = cross_val_score(model, X_train, Y_train_ln, cv=CV, scoring='neg_mean_squared_error', n_jobs=-1)

    rmse_log = np.sqrt(-scores.mean())
    return rmse_log
    # return np.mean(scores)  # Maximize score (negative RMSE or MAPE)

# Run Optuna Study
study = optuna.create_study(direction="minimize")  # or "minimize" based on your metric
study.optimize(objective, n_trials=N_ITER, n_jobs=-1)  # Increase n_trials for better results

# Get best hyperparameters
best_params = study.best_params
print("Best Hyperparameters:", best_params)
print("Best Score:", study.best_value)




import xgboost as xgb
import os
import joblib

if not os.path.exists(os.path.join(os.getcwd(), 'models', 'xgb_qo.pkl')) or True:
  xbg_qo =  xgb.XGBRegressor(tree_method="hist", objective='reg:quantileerror', quantile_alpha=0.5, **best_params)
  #xbg_qo =  xgb.XGBRegressor(tree_method="hist", objective='reg:pseudohubererror', **best_params)
  xbg_qo.fit(X_train[base_cols], Y_train_ln)

  joblib.dump(xbg_qo, 'models/xgb_qo.pkl')

xbg_qo = joblib.load('models/xgb_qo.pkl')

Y_predict_train = np.exp(xbg_qo.predict(X_train))
Y_predict_test = np.exp(xbg_qo.predict(X_test))

get_metrics(Y_train, Y_predict_train, 'Train')
results["XGBRegressorQuantile"] = get_metrics(Y_test, Y_predict_test, 'Test')   


params = xbg_qo.get_params()

filtered_params = {k: v for k, v in params.items() if v is not None}
print(filtered_params)
#filtered_params = dict(filter(lambda item: item[1] is not None, params.items()))




price_bins = [250000, 500000, 1000000, 1500000, 2000000, 3000000]
district_errors, price_errors = compute_errors(X_train, Y_train, Y_predict_train, X_test, Y_test, Y_predict_test, price_bins)




feature_importance = xbg_qo.feature_importances_
feature_names = X_train.columns

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False).reset_index(drop=True)


importance_df.head(20)




from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
import numpy as np

base_estimator = DecisionTreeRegressor(max_depth=3)

ada_boost_model = AdaBoostRegressor(estimator=base_estimator)

param_dist = {
    'n_estimators': np.arange(50, 400),
    'learning_rate': np.logspace(-3, 0, 10), #
    'estimator__max_depth': [3, 4, 5, 6, 7, 10, 12],
    'estimator__min_samples_split': np.arange(2, 21),
    'estimator__min_samples_leaf': np.arange(1, 21)
}


N_ITER = 50  # Number of parameter settings that are sampled
SCORING = 'neg_mean_absolute_error' #'neg_root_mean_squared_error' #'neg_mean_squared_error'  # Scoring metric
CV = 5  # Number of folds in cross-validation
RANDOM_STATE = 42

ada_boost_cv = RandomizedSearchCV(
    estimator=ada_boost_model,
    param_distributions=param_dist,
    n_iter=N_ITER,
    scoring=SCORING,
    cv=CV,
    n_jobs=-1,
    verbose=1,
    random_state=RANDOM_STATE,
    return_train_score=True
)

import joblib

if not os.path.exists(os.path.join(os.getcwd(), 'models', 'ada_boost_cv.pkl')):
  # Fit the model
  ada_boost_cv.fit(X_train, Y_train_ln)
  joblib.dump(ada_boost_cv, 'models/ada_boost_cv.pkl')

ada_boost_cv = joblib.load('models/ada_boost_cv.pkl')

# Get the best parameters
best_params = ada_boost_cv.best_params_
print("Best Parameters:", best_params)



import xgboost as xgb


Y_predict_train = np.exp(ada_boost_cv.predict(X_train))
Y_predict_test = np.exp(ada_boost_cv.predict(X_test))

get_metrics(Y_train, Y_predict_train, 'Train')

results["AdaBoostRegressor"] = get_metrics(Y_test, Y_predict_test, 'Test')


best_ada = ada_boost_cv.best_estimator_ #.named_steps['model']

feature_importance = best_ada.feature_importances_
feature_names = X_train.columns

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False).reset_index(drop=True)


importance_df.head(20)



price_bins = [250000, 500000, 750000, 1000000, 1250000, 1500000, 2000000, 3000000]
district_errors, price_errors = compute_errors(X_train, Y_train, Y_predict_train, X_test, Y_test, Y_predict_test, price_bins)




import lightgbm as lgb
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV


prefix = "model__"

lgbm_model = lgb.LGBMRegressor(alpha=0.5)

param_grid = {
    f"{prefix}objective": ["regression",  "quantile"], #"huber",
    f"{prefix}learning_rate": [0.01, 0.05], # , 0.1
    f"{prefix}n_estimators": [200, 500], #, 1000
    f"{prefix}num_leaves": [31, 50],
    f"{prefix}max_depth": [-1, 10],
    f"{prefix}min_child_samples": [10, 20],
    f"{prefix}subsample": [0.7, 0.9, 1.0],
    f"{prefix}colsample_bytree": [0.7, 0.9, 1.0]
}

lgbm_q = GridSearchCV(
    estimator=get_pipeline(lgbm_model),
    param_grid=param_grid,
    scoring=SCORING,
    cv=CV,
    n_jobs=-1,
    verbose=1
)

lgbm_q.fit(X_train, Y_train_ln)

lgbm_q.best_params_




import xgboost as xgb


Y_predict_train = np.exp(lgbm_q.predict(X_train))
Y_predict_test = np.exp(lgbm_q.predict(X_test))

get_metrics(Y_train, Y_predict_train, 'Train')
results["LGBMRegressor"] = get_metrics(Y_test, Y_predict_test, 'Test')



import matplotlib.pyplot as plt


plt.figure(figsize=(10, 6))

plt.scatter(Y_predict_train, Y_train, color="red", alpha=0.05)
plt.scatter(Y_predict_test, Y_test, color="blue", alpha=0.05)

plt.title('Predicted vs Real price')
plt.xlabel('Predicted')
plt.ylabel('Real')
plt.legend()

plt.show()


best_xgb = xbg_q.best_estimator_.named_steps['model']
feature_importance = best_xgb.feature_importances_
feature_names = X_train.columns

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False).reset_index(drop=True)


importance_df.head(10)


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import numpy as np


param_dist = {
    f'{prefix}n_estimators': randint(50, 500),
    f'{prefix}max_depth': randint(1, 10),
    f'{prefix}min_samples_split': randint(2, 20),
    f'{prefix}min_samples_leaf': randint(1, 5),
    f'{prefix}max_features': ['sqrt', 'log2', None],
    f'{prefix}bootstrap': [True, False],
    f'{prefix}criterion': ['friedman_mse', 'squared_error', 'poisson'], #{'friedman_mse', 'squared_error', 'absolute_error', 'poisson'}
}


rf = get_pipeline(RandomForestRegressor(random_state=RANDOM_STATE))


random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=N_ITER,
    cv=CV,
    scoring=SCORING,
    n_jobs=-1,
    verbose=1,
    random_state=RANDOM_STATE,
    return_train_score=True
)


random_search.fit(X_train, Y_train)

print("Best parameters found:")
print(random_search.best_params_)
print(f"\nBest cross-validation score: {random_search.best_score_:.3f}")

# Get the best model
best_rf = random_search.best_estimator_



Y_predict_train = best_rf.predict(X_train)
Y_predict_test = best_rf.predict(X_test)

get_metrics(Y_train, Y_predict_train, 'Train')
results["RandomForestRegressor"] = get_metrics(Y_test, Y_predict_test, 'Test')  




## Add final comparison table for all methods
pd.set_option('display.float_format', '{:.4f}'.format)
combined_results = pd.concat(
    {model: df.T for model, df in results.items()},
    axis=1
)

combined_results.columns = combined_results.columns.droplevel(1)

combined_results