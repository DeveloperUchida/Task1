import matplotlib.pyplot as plt
import sklearn.datasets

digits = sklearn.datasets.load_digits()
for i in range(50):
    plt.subplot(5,10,i + 1)
    plt.axis("OFF")
    plt.title(digits.target[i])
    plt.imshow(digits.images[i], cmap="Greys")
plt.show()
