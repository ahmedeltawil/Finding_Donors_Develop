import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
import visuals as vs
from IPython.display import display
# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler
# Import train_test_split
from sklearn.model_selection import train_test_split



data=pd.read_csv("census.csv")

# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))


# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

# Show an example of a record with scaling applied
#print(features_log_minmax_transform.head(n = 5))

features_final=pd.get_dummies(features_log_minmax_transform, prefix=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'])


d = {data['income'].unique()[0]: 0, data['income'].unique()[1]: 1}
income = income_raw.map({data['income'].unique()[0]: 0, data['income'].unique()[1]: 1})
#print(income)


# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final,
                                                    income,
                                                    test_size = 0.2,
                                                    random_state = 0)

print(len(y_train))
print(round(len(y_train)*0.1))