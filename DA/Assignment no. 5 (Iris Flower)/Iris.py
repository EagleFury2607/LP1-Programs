
"""**Load Libraries**"""

import pandas
import matplotlib.pyplot as plt

"""**Load Dataset**"""

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

"""**Shape**"""

print(dataset.shape)

"""**Head**"""

print(dataset.head(20))

"""**Descriptions**"""

print(dataset.describe())

"""**Class Distribution**"""

print(dataset.groupby('class').size())

"""**Histograms **"""

dataset.hist()
plt.show()

"""**Box Plot**"""

dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

