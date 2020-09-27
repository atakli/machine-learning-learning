### İRİS İÇİN SINIFLANDIRMA
import seaborn as sns
iris = sns.load_dataset('iris')
iris.head()
X_iris = iris.drop('species', axis=1)
y_iris = iris['species']
from sklearn.model_selection import train_test_split
X_egitim, X_test, y_egitim, y_test = train_test_split(X_iris, y_iris, random_state=1)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_egitim, y_egitim)
y_model = model.predict(X_test)
from sklearn.metrics import accuracy_score # buna ne gerek var anlamadım. model.score(X_test,y_test) de işi görüyo aslında
accuracy_score(y_test, y_model)	
# model.score(X_egitim,y_egitim): 0.9464285714285714
# model.score(X_test,y_test): 0.9736842105263158

################################## MODEL İŞLEMLERİ
import matplotlib.pyplot as plt
__import__('matplotlib').interactive(True)
import numpy as np
rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)
plt.scatter(x,y)
# bu grafikten gördük ki x ile y arasında linear bir ilişki var, ki zaten denklemden onu bekliyorduk
from sklearn.linear_model import LinearRegression
# belirleyeceğimiz parametrelere hiper parametre denir. onları yazmazsak default parametreler geçerli olur
model = LinearRegression(fit_intercept=True) # fit_intercept zaten True. neden yazdık anlamadım. belki tutorial'daki sürüm farklıdır
X = x[:,np.newaxis]
model.fit(X,y) 
model.coef_ # sklearn'de işler böyle gibi bişey söyledi (alt-tire hakkında)
model.intercept_ # (modelin sabiti)
# model kurulduktan sonraki adım yeni verileri değerlendirmektir. bunun için predict metodunu kullanıcaz
x_fit = np.linspace(-1,11) # örnek bir girdi. -1 ile 11'i x.min ile x.max'a göre belirleyebiliriz
X_fit = x_fit[:,np.newaxis] # predict metodu tek boyutlu arrayleri kabul etmiyo demekki
y_fit = model.predict(X_fit)
plt.figure()
plt.scatter(x,y)
plt.plot(x_fit,y_fit)