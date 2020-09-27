# regresyonda amaç sürekli değerleri tahmin etmektir
# bir kişinin eğitim, yaş gibi verileriyle bir neticeyi tahmin etmek regresyana örnektir
# sonuç değişkeni sürekli mi kategorik mi: regression or classification
# k-en yakın komşu algoritması en basit ml'lerden biridir
# mglearn kütüphanesi ml'yi öğretmek için yazılmış
# coding: utf-8
import mglearn
mglearn.plots.plot_knn_classification(n_neighbors=1) # bi de 3 yazdı
from sklearn.model_selection import train_test_split
X,y = mglearn.datasets.make_forge()
X_egitim, X_test, y_egitim, y_test = train_test_split(X,y,random_state=0)
from sklearn.neighbors import KNeighborsClassifier
snf = KNeighborsClassifier(n_neighbors=3)
snf.fit(X_egitim,y_egitim)
snf.predict(X_test)
snf.score(X_test,y_test) # bu score metodu hem predict yapıyo hem de mean. # 0.8571428571428571
snf.score(X_egitim,y_egitim) # 0.9473684210526315
### Gerçek veri seti ile uygulama
from sklearn.datasets import load_breast_cancer
kanser = load_breast_cancer()
kanser.keys()
print(kanser.DESCR)
X_egitim, X_test, y_egitim, y_test = train_test_split(kanser.data, kanser.target, stratify=kanser.target ,random_state=66)
egitim_dogruluk = []
test_dogruluk = []
komsuluk_sayisi = range(1,11)
for n_komsuluk in komsuluk_sayisi:
    snf = KNeighborsClassifier(n_neighbors=n_komsuluk)
    snf.fit(X_egitim,y_egitim)
    egitim_dogruluk.append(snf.score(X_egitim,y_egitim))
    test_dogruluk.append(snf.score(X_test,y_test))
plt.plot(komsuluk_sayisi, egitim_dogruluk, label='Eğitim doğruluk')
plt.plot(komsuluk_sayisi, test_dogruluk, label='Test doğruluk')
plt.ylabel('Doğruluk')
plt.xlabel('n-komşuluk')
plt.legend()
# az komşuluk sayısıyla kurulan model kompleks modeldir ama bu modelin doğruluk oranları düşüktür
# çok komşuluk sayısıyla kurulan model basit modeldir ama bu modelin de performansı kötüdür
# ???
# komşuluk sayılarına göre kurulan modellerde modelin performansı ve doğruıluk oranı birlikte düşünüldüğünde en
# iyi model grafikte görüldüğü gibi yaklaşık 6 komşuluk sayısına göre oluşturulan modeldir
### Regresyon (k- en yakın komşuluğu'nun regresyon çeşidi)
mglearn.plots.plot_knn_regression(n_neighbors=1) # # bi de 3 yazdı
from sklearn.neighbors import KNeighborsRegressor
X,y = mglearn.datasets.make_wave(n_samples=40)
X_egitim, X_test, y_egitim, y_test = train_test_split(X,y,random_state=0)
reg = KNeighborsRegressor(n_neighbors=3)
reg.fit(X_egitim,y_egitim)
reg.score(X_test,y_test)
