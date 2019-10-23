import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

s = []
s.append(1)
s.append(0)
s.append(1)
s.append(0)
s.append(2)

encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(s)
# convert integers to dummy variables (one hot encoding)
dummy_y = np_utils.to_categorical(encoded_Y)

print(dummy_y)
r = []
for val in dummy_y:
    r.append(np.argmax(val))

print(r)
