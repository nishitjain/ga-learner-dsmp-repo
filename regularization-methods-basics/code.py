# --------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# path- variable storing file path
df = pd.read_csv(path)

#Code starts here
print(df.head())
X = df.drop(['Price'],1)
y = df.Price
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=6)
corr = X_train.corr()
print(corr)



# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Code starts here
regressor = LinearRegression()
y_pred = regressor.fit(X_train,y_train).predict(X_test)
r2 = r2_score(y_test,y_pred)
print('R2 Score: {}'.format(r2))


# --------------
from sklearn.linear_model import Lasso

# Code starts here
lasso = Lasso()
lasso_pred = lasso.fit(X_train,y_train).predict(X_test)
r2_lasso = r2_score(y_test,lasso_pred)
print('Lasso R2 Score: {}'.format(r2_lasso))


# --------------
from sklearn.linear_model import Ridge

# Code starts here
ridge = Ridge()
ridge_pred = ridge.fit(X_train,y_train).predict(X_test)
r2_ridge = r2_score(y_test,ridge_pred)
print('Ridge R2 Score: {}'.format(r2_ridge))
# Code ends here


# --------------
from sklearn.model_selection import cross_val_score

#Code starts here
regressor = LinearRegression()
score = cross_val_score(regressor,X_train,y_train,cv=10)
print('Cross Validation Scores: {}'.format(score))
mean_score = score.mean()
print('Mean Cross Validation Score: {}'.format(mean_score))


# --------------
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#Code starts here
model = make_pipeline(PolynomialFeatures(2),LinearRegression())
y_pred = model.fit(X_train,y_train).predict(X_test)
r2_poly = r2_score(y_test,y_pred)
print('Polynomial Regression R2 Score: {}'.format(r2_poly))


