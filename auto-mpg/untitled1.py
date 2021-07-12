import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, skew
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import train_test_split, GridSearchCV
#XGBoost
import xgboost as xgb

#warnings
import warnings
warnings.filterwarnings('ignore')

columns_name=['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'weight', 'Acceleration', 'Model Year', 'Origin']

data=pd.read_csv('C:\\Users\\natiq\\Desktop\\auto-mpg\\auto-mpg.data', names=columns_name, na_values="?", comment='\t',
                sep=' ', skipinitialspace=True)


data=data.rename(columns={'MPG':'target'})

describe=data.describe()

data.head()

data['Horsepower']=data['Horsepower'].fillna(data['Horsepower'].mean())

#EDA

corr_matrix=data.corr()

sns.clustermap(corr_matrix, annot=True, fmt='.2f')
plt.title('Correlation btw features')
plt.show()

threshold=0.75

filtre=np.abs(corr_matrix['target'])>threshold

corr_features=corr_matrix.columns[filtre].tolist()

sns.clustermap(data[corr_features].corr(), annot=True, fmt='.2f')
plt.title('Correlation btw features')
plt.show()

#Multi-culinarity
#Bir birileri ile birden cok iliskili
#5 feature eyni seyi deyir, biri besdir


sns.pairplot(data, diag_kind='kde', markers='+')

#box

for c in data.columns:
    plt.figure()
    sns.boxplot(x=c, data=data, orient='v')
    
#Outlier

thr=2

horsepower_desc=describe['Horsepower']

Q3=horsepower_desc[6]

Q1=horsepower_desc[4]

IQR_hp=Q3-Q1

top_limit=Q3+ thr*IQR_hp

bottom_limit=Q1- thr*IQR_hp

filter_hp_bottom=bottom_limit<data['Horsepower']

filter_hp_top=data['Horsepower']<top_limit

filter_hp=filter_hp_bottom & filter_hp_top

data=data[filter_hp]

#Eyni sey Acceleration

horsepower_desc=describe['Acceleration']

Q3=horsepower_desc[6]

Q1=horsepower_desc[4]

IQR_hp=Q3-Q1

top_limit=Q3+ thr*IQR_hp

bottom_limit=Q1- thr*IQR_hp

filter_hp_bottom=bottom_limit<data['Acceleration']

filter_hp_top=data['Acceleration']<top_limit

filter_hp=filter_hp_bottom & filter_hp_top

data=data[filter_hp]

#Feature Engineering

#Skewness

#target dependent variable

sns.distplot(data.target)

sns.distplot(data.target, fit=norm)

(mu, sigma)=norm.fit(data['target'])

print(mu, sigma)

#qq plot
fig=plt.figure()
stats.probplot(data['target'], plot=plt)
plt.show()

data['target']=np.log1p(data['target'])

sns.distplot(data.target, fit=norm)
(mu, sigma)=norm.fit(data['target'])

print(mu, sigma)

fig=plt.figure()
stats.probplot(data['target'], plot=plt)
plt.show()

#feature- independent variable

skewed_feats=data.apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness=pd.DataFrame(skewed_feats, columns=['skewned'])

#Box Cox Transformation lazim olsa skew ucun

#One Hot encoding

data['Cylinders']=data['Cylinders'].astype(str)

data['Origin']=data['Origin'].astype(str)

data=pd.get_dummies(data)

#Split and stand

X=data.drop(['target'], axis=1)

Y=data.target

test_size=90
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=test_size, random_state=42)

scaler=StandardScaler() #RobustScaler

#mean=0, std=1
X_train=scaler.fit_transform(X_train)

X_test=scaler.fit_transform(X_test)

#Linear Regression
lr=LinearRegression()

lr.fit(X_train, Y_train)

print(lr.coef_)

Y_pred=lr.predict(X_test)

mse=mean_squared_error(Y_test, Y_pred)

print(mse)

#Ridge- over onluyur, varyansi azaldir

ridge=Ridge(random_state=42, max_iter=10000)

alphas=np.logspace(-4, -0.5, 30)

tuned_params=[{'alpha':alphas}]

n_folds=5


clf=GridSearchCV(ridge, tuned_params, cv=n_folds,
                 scoring='neg_mean_squared_error', refit=True)

clf.fit(X_train, Y_train)

scores=clf.cv_results_['mean_test_score']

scores_std=clf.cv_results_['std_test_score']

print(clf.best_estimator_.coef_)

ridge=clf.best_estimator_

print(ridge)

Y_pred=clf.predict(X_test)

mse=mean_squared_error(Y_test, Y_pred)

print(mse)

plt.semilogx(alphas, scores)

#########Lasso

#katsayi dusuk olanlari 0 edir

#Lasso-Feature selection
#high corr varsa birini saxlayir, digerleri 0 olur
#bir az bias olur/ amma var azalir


lasso=Lasso(random_state=42, max_iter=10000)

alphas=np.logspace(-4, -0.5, 30)

tuned_params=[{'alpha':alphas}]

n_folds=5


clf=GridSearchCV(lasso, tuned_params, cv=n_folds,
                 scoring='neg_mean_squared_error', refit=True)

clf.fit(X_train, Y_train)

scores=clf.cv_results_['mean_test_score']

scores_std=clf.cv_results_['std_test_score']

print(clf.best_estimator_.coef_)

ridge=clf.best_estimator_

print(ridge)

Y_pred=clf.predict(X_test)

mse=mean_squared_error(Y_test, Y_pred)

print(mse)

plt.semilogx(alphas, scores)

#ElasticNet

#High corr-da azalma ve cixarma edir
alphas=np.logspace(-4, -0.5, 30)


eNet=ElasticNet(random_state=42, max_iter=10000)

tuned_params={'alpha':alphas, 'l1_ratio':np.arange(0.0, 1.0, 0.05)}

n_folds=5


clf=GridSearchCV(eNet, tuned_params, cv=n_folds,
                 scoring='neg_mean_squared_error', refit=True)

clf.fit(X_train, Y_train)

scores=clf.cv_results_['mean_test_score']

scores_std=clf.cv_results_['std_test_score']

print(clf.best_estimator_.coef_)

eNet=clf.best_estimator_

print(eNet)

Y_pred=clf.predict(X_test)

mse=mean_squared_error(Y_test, Y_pred)

print(mse)

####XGBoost

#RobustScaler()-outlier-lari veriden uzaklasdirir

model_xgb=xgb.XGBRegressor(objective='reg:linear', max_depth=5,
                           min_child_weight=4, subsample=0.7,
                           n_estimators=100, learning_rate=0.7)

model_xgb.fit(X_train, Y_train)

Y_pred=model_xgb.predict(X_test)

mse=mean_squared_error(Y_test, Y_pred)

print(mse)

###########################

params={'nthread':[4],
        'objective':['reg:linear'],
        'learning_rate':[.03, 0.05, .07],
        'max_depth':[5, 6, 7],
        'min_child_weight':[4],
        'silent':[1],
        'subsample':[0.7],
        'colsample_bytree':[0.7],
        'n_estimators':[500, 1000]
        }

model_xgb=xgb.XGBRegressor()

clf=GridSearchCV(model_xgb, params, cv=n_folds,
                 scoring='neg_mean_squared_error',
                 refit=True, n_jobs=5)

model_xgb=clf.best_estimator_

print(model_xgb)

clf.fit(X_train, Y_train)

Y_pred=clf.predict(X_test)

mse=mean_squared_error(Y_test, Y_pred)

print(mse)

#Averaging models

class AM():
    
    def __init__(self, models):
        self.models=models
        
    def fit(self, X, Y):
        self.models=[clone(x) for x in self.models]

        for m in self.models:
            m.fit(X, Y)
            
        return self
    
    def predict(self, X):
        
        predictions=np.column_stack(
            [m.predict(X) for m in self.models])


        return np.mean(predictions, axis=1)


a=AM(models=(model_xgb, lasso))
a.fit(X_train, Y_train)

Y_pred=a.predict(X_test)

mse=mean_squared_error(Y_test, Y_pred)

print(mse)






















