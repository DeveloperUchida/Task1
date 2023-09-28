import PIL.Image
import numpy
import sklearn.datasets
import sklearn.svm


# 画像ファイルを数値リストに変換する
def imageToData(filename):
    # 画像を8x8のグレースケールに変換
    grayImage = PIL.Image.open(filename).convert("L")
    grayImage = grayImage.resize((8, 8), PIL.Image.Resampling.LANCZOS)

    # 数値リストに変換
    numImage = numpy.asarray(grayImage, dtype=float)
    numImage = 16 - numpy.floor(17 * numImage / 256)
    numImage = numImage.flatten()

    return numImage


# 数字を予測する
def predictDigits(data):
    # 学習用データを読み込む
    digits = sklearn.datasets.load_digits()

    # 機械学習をする
    clf = sklearn.svm.SVC(gamma=0.01)
    clf.fit(digits.data, digits.target)
