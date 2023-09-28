import matplotlib.pyplot as plt
import sklearn.datasets

digits = sklearn.datasets.load_digits()
plt.imshow(digits.images[0], cmap="Greys")
plt.show()
