import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


# load dataset
dataframe = pd.read_csv("./data/iris.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]

# encode class values as integers
encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(Y)
# convert integers to dummy variables (one hot encoding)
dummy_y = np_utils.to_categorical(encoded_Y)

# define model structure
seed = 42
np.random.seed(seed)


def baseline_model():
    model = Sequential()
    model.add(Dense(units=4, input_dim=4, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(units=6, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(units=3, activation='softmax', kernel_initializer='glorot_uniform'))
    # model.add(Dropout(0.2))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
# splitting data into training set and test set. If random_state is set to an integer, the split datasets are fixed.
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.2, random_state=0)  # test_size = 0.3
estimator.fit(X_train, Y_train)

# make predictions
pred = estimator.predict(X_test)
print(pred)
print(Y_test)

# predicted_label = estimator.predict_classes(X_test)
# print(predicted_label)
# print(pred)
# print(Y_test)
# print(Y_train)
# print(Y_test)
# print("$$$$$$$$$$$$$$$$$$")
# inverse numeric variables to initial categorical labels
# init_lables = encoder.inverse_transform(pred)
# print(init_lables)

# k-fold cross-validate
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, dummy_y, cv=kfold)
results = cross_val_score(estimator, X_test, Y_test, cv=kfold)
print('Accuracy: %.2f%% (%.2f)' % (results.mean() * 100, results.std()))
