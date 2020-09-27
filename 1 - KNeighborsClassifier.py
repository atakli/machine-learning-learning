# coding: utf-8
from sklearn.datasets import load_iris 
iris = load_iris() 
iris.keys()
# print(iris['DESCR']) # veri setinin özet bilgileri
iris['target_names']
iris.keys()
iris['feature_names']
type(iris['data'])
"""
iris = sns.load_dataset('iris')
X_iris = iris.drop('species', axis=1)
y_iris = iris['species']
burdaki X_iris pandas dataframe. ama yine de np.array_equal(iris_sk.data,X_iris) True verdi (iris_sk sklearn'den gelen iris)
"""
iris['data'].shape
# ilk değeri örneklem sayısı, ikinci değeri nitelik sayısını gösterir, sckitlearn'de
iris['data'][:5]
iris['target'] # herbir çiçeğin tipi
# model iki parçaya bölünür: training verisi ve test verisi
from sklearn.model_selection import train_test_split # satırları karıştırıp %75'ini train, gerisini test verisi olarak ayırır
X_eğitim, X_test, y_eğitim, y_test = train_test_split(iris['data'],iris['target'],random_state=0)
X_eğitim.shape
y_eğitim.shape
X_test.shape
y_test.shape
# veride acayiplikler olabilir. mesela bazı uzunluklar m bazıları inch olabilir.
# bunun için görselleştirmek iyi bi yol: saçılım grafiği
import pandas as pd
iris_df = pd.DataFrame(X_eğitim, columns = iris.feature_names)
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
__import__('matplotlib').interactive(True)
scatter_matrix(iris_df,c=y_eğitim,figsize=(15,15),marker='o',hist_kwds={'bins':20},s=80,alpha=0.8)
# c: verinin türlere göre renklenmesi için
# hist_kwds: histogramların dikdörtgen genişiliği
# s: noktaların büyüklüğü
# alpha: noktaların görünümü

# bu grafikten spal ve petal özelliklerine göre üç sınıfın güzel ayrılmış olduğunu görebiliyoruz. dolayısıyla, yazacağımız makine 
# öğrenmesi modeli büyük ihtimalle bu sınıfları ayırmayı iyi öğrenecek
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_eğitim,y_eğitim)
import numpy as np
X_yeni = np.array([5,2.9,1,0.2])
X_yeni.shape
# sklearn her zaman verinin iki boyutlu olmasını bekler
X_yeni = np.array([[5,2.9,1,0.2]])	# or X_yeni = X_yeni[np.newaxis,:]
tahmin = knn.predict(X_yeni)
print('Tahmin sınıfı: ',tahmin)
print('Tahmin türü: ',iris['target_names'][tahmin])
knn.score(X_test,y_test)	
# 0.9736842105263158 # veya:
y_tahmin = knn.predict(X_test)
np.mean(y_tahmin == y_test)
