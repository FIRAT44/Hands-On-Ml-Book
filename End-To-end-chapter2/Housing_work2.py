import pandas as pd
import os 
import numpy as np

HOUSING_PATH = os.path.join("datasets", "housing")
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


housing = load_housing_data()
print(housing.head())

housing["income_cat"] = pd.cut(housing["median_income"],
bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
labels=[1, 2, 3, 4, 5])

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# Data Cleaning
housing.dropna(subset=["total_bedrooms"]) # option 1
housing.drop("total_bedrooms", axis=1) # option 2
median = housing["total_bedrooms"].median() # option 3
housing["total_bedrooms"].fillna(median, inplace=True)


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity",axis=1) 
imputer.fit(housing_num)

print("\n----------------------------------\n----------------------------------\n")
print(imputer.statistics_)
print("\n----------------------------------\n----------------------------------\n")

print(housing_num.median().values)
print("\n----------------------------------\n----------------------------------\n")
X= imputer.transform(housing_num)

housing_tr  = pd.DataFrame(X,columns=housing_num.columns)

# Handling Text and Categorical Attributes
housing_cat = housing[["ocean_proximity"]]
print("\n----------------------------------\n")
print(housing_cat.head(10))

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()

housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print("\n----------------------------------\n")
print(housing_cat_encoded[:10])
print("\n----------------------------------\n")
print(ordinal_encoder.categories_)
print("\n----------------------------------\n")

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print(housing_cat_1hot.toarray())
print("\n----------------------------------\n")
print(cat_encoder.categories_)
print("\n----------------------------------\n")