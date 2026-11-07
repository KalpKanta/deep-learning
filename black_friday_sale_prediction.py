import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

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
    - set(id_cols)
    - set(target_col)
    - set(category_col)
    - set(flag_col)
)

for var in category_col + number_col:
    if train_test[var].isnull().any():
        train_test[var+"_NA"] = train_test[var].isnull()*1