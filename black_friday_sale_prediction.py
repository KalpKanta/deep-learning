import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStoping
from sklearn.metrics import mean_squared_error
from tensorflow.keras.regularizers import l2

train = pd.read_csv("train_black_friday.csv")
test = pd.read_csv("test_black_friday.csv")
print(train["Product_ID"])

train["Product_ID"] = train["Product_ID"].astype("category").cat.codes
test["Product_ID"] = test["Product_ID"].astype("category").cat.codes
train["type"] = "train"
test["type"] = "test"
train_test = pd.concat([train, test], access = 0)
id_col = ["User_ID", "Product_ID"]
flag_col = ["type"]
target_col = ["Purchase"]
category_col = ["Gender","Age","Occupation","City_Category","Stay_In_Current_City_Years"]
number_col = list(
    set(train_test.columns)
    - set(id_col)
    - set(target_col)
    - set(category_col)
    - set(flag_col)
)

for var in category_col + number_col:
    if train_test[var].isnull().any():
        train_test[var+"_NA"] = train_test[var].isnull()*1

train_test[number_col] = train_test[number_col].fillna(train_test[number_col].mean())
train_test[category_col] = train_test[categorl_col].fillna(-9999)

for var in category_col:
    le = LabelEncoder()
    train_test[var] = le.fit_transform(train_test[var].astype(str))

features = list(
    set(train_test.columns) 
    - set(id_cols)
    - set(target_col)
    - set(flag_col)
)

train_test[features] = train_test[features] / train_test[features].max()
train_data = train_test[train_test["type"] == "train"]
test_data = train_test[train_test["type"] == "test"]

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X = ss.fit_transform(train_data[features].values)
y = train_data[target_col].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

model = RandomForestRegressor(random_state = 42, n_estimators = 100)

model.fit(X_train,y_train.ravel())
y_pred = model.predict(X_test)
rmse_val = np.sqrt(mean_squared_error(y_test, y_predicted))
print(f"Validation RMSE neural network : {rmse_val}")

model= Sequential([
    Input(shape = (len(features),)),
    Dense(200, activation = "relu", kernel_regularizer = l2(0.01)),
    Dropout(0.3), 
    Dense(100, activation = "relu", kernel_regularizer = l2(0.01)),
    Dropout(0.3),
    Dense(50, activation = "relu"),
    Dense(1)
]) 

model.compile(optimizer = Adam(learning_rate = 0.001), loss = "mean_squared_error", metrics = ["mean_squared_error"])

es = EarlyStopping(monitor = "val_loss", patience = 3, restore_best_weights = True)
history = model.fit(X_train, y_train, epochs = 10, validation_data = (X_test, y_test), callbacks = es, verbose = 1)

y_pred = model.predict(X_test)
rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Validation RMSE sequential : {rmse_val}")

