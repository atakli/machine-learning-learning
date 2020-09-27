### KATEGORİK ÖZNİTELİKLER
data = [{'not':85,'kardes':4,'ders':'mat'},
		{'not':70,'kardes':3,'ders':'ing'},
		{'not':65,'kardes':3,'ders':'mat'},
		{'not':60,'kardes':2,'ders':'fiz'}]
from sklearn.feature_extraction import DictVectorizer
vek = DictVectorizer(sparse=False, dtype=int)
vek.fit_transform(data)
vek.get_feature_names()
vek
# sparse'ı True yapınca noluyo anlamadım. edit: anladım. output yazmıyo, sparse matrix oluşturuyo. o ne ?

### TEXT OZNITELIKLER
veri = ['hava iyi', 'iyi insan', 'hava bozuk']
from sklearn.feature_extraction.text import CountVectorizer
vek = CountVectorizer()
X = vek.fit_transform(veri)
X # böylece herbir kelimenin sayısı sparse olarak ekrana yazıldı
import pandas as pd
pd.DataFrame(X.toarray(),columns=vek.get_feature_names())

### OZNITELIK TURETME
import numpy as np
import matplotlib.pyplot as plt
x = np.array([1,2,3,4,5])
y = np.array([5,3,1,2,7])
__import__('matplotlib').interactive(True)
plt.scatter(x,y)
from sklearn.linear_model import LinearRegression
X = x[:,np.newaxis]
model = LinearRegression().fit(X,y)
y_fit = model.predict(X)
plt.scatter(x,y)
plt.plot(x,y_fit)
# x ile y arasındaki ilişkiyi tanımayabilmek için daha gelişmiş bir modele ihtiyaç olduğunu grafikten anladık
# veriyi dönüştürerek modeli daha iyi hale getirebiliriz
# böylece ekstra öznitelik sütunu ekleyerek modele esneklik kazandırabiliriz 
from sklearn.preprocessing import PolynomialFeatures
pol = PolynomialFeatures(degree=3, include_bias=False)
X2 = pol.fit_transform(X)
X2
model = LinearRegression().fit(X2,y)
y_fit = model.predict(X2)
plt.figure()
plt.scatter(x,y)
plt.plot(x,y_fit)
# şimdi model verilere daha iyi uyuyor. modeli iyileştirmiş olduk

### KAYIP (EKSİK) VERİLER
from numpy import nan
X = np.array([[1,nan,3],
             [5,6,9],
             [4,5,2],
             [4,6,nan],
             [9,8,1]])
y = np.array([10,13,-2,7,-6])
# from sklearn.preprocessing import Imputer # tutorial'da böyleydi ama module docs'tan bakınca bende aşağıdaki gibi olduğunu gördüm
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean')
X2 = imp.fit_transform(X)
X2
model = LinearRegression().fit(X2,y)
model.predict(X2)

### PIPELINE TEKNİĞİ
from sklearn.pipeline import make_pipeline
model = make_pipeline(SimpleImputer(strategy='mean'),PolynomialFeatures(degree=2),LinearRegression())
# make_pipeline sadece satır azaltmak için mi
# PolynomialFeatures özelliği yukarıda görüldüğü gibi daha precise sonuç veriyo
model.fit(X,y)
y_fit = model.predict(X)

