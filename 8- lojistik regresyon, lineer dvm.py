# Ridge denilen L2 regülerleştirmesi her iki modelde de (LinearSVC ve LogisticRegression) öntanımlı olarak uygulanır
# modedec argümanı regülerleştirme baskısını belirler
# c'nin yüksek değeri daha az regülerleştirme yapar
# c'nin düşük değeri katsayıları 0'a yakınlaştırmak için baskı yapar, dolayısıyla daha çok regülerleştirme olur
# c'nin 100 olduğu model bütün noktaları doğru sınıflandırmak için büyük çaba harcıyor ama muhtemelen bu modelde
# overfitting vardır

### LOJİSTİK REGRESYON
from sklearn.datasets import load_breast_cancer
kanser = load_breast_cancer()
from sklearn.model_selection import train_test_split
X_egitim, X_test, y_egitim, y_test = train_test_split(kanser.data,kanser.target,stratify=kanser.target,random_state=42)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression().fit(X_egitim,y_egitim) # bunda uyarı verdi.
logreg = LogisticRegression(solver='liblinear').fit(X_egitim,y_egitim)
# eğer veri seti yüzbinlerce veya milyonlarca örnekten oluşuyorsa solver argümanına "sa" (?) yazabiliriz
print(logreg.score(X_egitim,y_egitim))	# 0.9553990610328639
print(logreg.score(X_test,y_test))		# 0.951048951048951
# bu değerler birbirine çok yakın olduğu için muhtemelen modelde underfitting var
# bu durumun üstesinden gelmek için c'yi 100 yapalım
logreg100 = LogisticRegression(C=100,solver='liblinear').fit(X_egitim,y_egitim)
print(logreg100.score(X_egitim,y_egitim))	# 0.9671361502347418
print(logreg100.score(X_test,y_test))		# 0.965034965034965
# eğitim ve test verisindeki performansı yükseldi, C'nin değerini yükselttiğimizde model daha iyi çalıştı
# C'nin değerini büyüterek regülerleştirmeyi azalttık
# böylece modelin eğitim verisindeli doğruluğu arttı, test verisinde ise hafif bir artış oldu (ikisinde de hafif bi artış oldu)
# (aslında eğitim skoru tutorial'da 0.9765...) # birkaç gün sonra kendimde tekrar çalıştırdım, bende de farklı oldu.
# bi de C'yi düşürelim
logreg001 = LogisticRegression(C=0.01,solver='liblinear').fit(X_egitim,y_egitim) 
print(logreg001.score(X_egitim,y_egitim))	# 0.9342723004694836
print(logreg001.score(X_test,y_test))		# 0.9300699300699301
# regülerleştirme argümanını azalttığımız zaman beklediğimiz gibi hem eğitim hem test doğruluk değerleri düştü
# şimdi C'nin aldığı değerlere göre modellerin öğrendiği katsayıları grafikte görelim:
# (grafiğin kodunu göstermedi)
# C=0.001 durumunda regülerleştirme arttığı için çoğu katsayı 0'a oldukça yaklaşmış. C arttıkça regülerleştirme azalıyor
for C, marker in zip([0.01,1,100],['o','^','v']):
    lr_l1 = LogisticRegression(penalty='l1',max_iter=1000,solver='liblinear',C=C).fit(X_egitim,y_egitim)
	# max_iter'i yazmayınca uyarı veriyo. arttırmamız isteniyo
    print('C={:.3f} için eğitim doğruluk {:.2f}'.format(C,lr_l1.score(X_egitim,y_egitim)))
    print('C={:.3f} için test doğruluk {:.2f}'.format(C,lr_l1.score(X_test,y_test)))
# C=0.010 için eğitim doğruluk 0.92
# C=0.010 için test doğruluk 0.93
# C=1.000 için eğitim doğruluk 0.96
# C=1.000 için test doğruluk 0.96
# C=100.000 için eğitim doğruluk 0.99
# C=100.000 için test doğruluk 0.98
# C'nin 0.01 değeri için regülerleştirmenin arttığını ve modelin Lineer Regresyona yaklaştığını görüyoruz
##### edit: SORU: önceki (7.) tutorial metninde tam tersini söylemişiz ?

### ÇOKLU SINIFLANDIRMA
# birçok lineer sınıflandırma modeli ikili sınıfladırma için kullanılır. 
# Logistic Regresyon hariç ikili sınıflandırma direk olarak çok kategorili sınıflandırmaya genişletilemez
# bu genişletme için ban-rez tekniği kullanılır
# bu teknikte herbir sınıf diğer bütün sınıflardan ayrılmayı dener
# 3 kategorili bir veri seti için bu tekniği uygulayalım
from sklearn.datasets import make_blobs
import mglearn
X,y = make_blobs(random_state=42)
__import__('matplotlib').interactive(True)
mglearn.discrete_scatter(X[,:0],X[,:1],y)
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.xlabel('Öznitelik 0')
import matplotlib.pyplot as plt
plt.xlabel('Öznitelik 0')
plt.ylabel('Öznitelik 1')
plt.legend(['sınıf 0','sınıf 1','sınıf 2'])
### LİNEER DVM
# dvm'Nin birçok modeli var, burda sadece lineer olanına bakıcaz
from sklearn.svm import LinearSVC
linear_svm = LinearSVC().fit(X,y)
linear_svm.coef_
# bu dizideki herbir satır sınıfların katsayı vektörlerini, 
# sütunlar da veri setinden aldığımız özniteliklerin katsayı değerlerini gösteriyo
X.shape
y.shape
# galiba evet, X'in shape'ine bakınca 2 tane öznitelik var gibi duruyo. iki tane sütun olması bu yüzden demekki. 
# üç satır olması da üç sınıf olduğu için, allahu alem
# heralde make_blobs denen veri seti üç kategorili, yani üç sınıflı
get_ipython().run_line_magic('pinfo', 'make_blobs') # burda aslında ?make_blobs yazmıştım sadece
# help'e bakınca anlaşıldı: öznitelik ve feature sayısı changable
# şimdi sınıfları ikiye parçalayan üç doğrunun grafiğini çizelim
import numpy as np
mglearn.discrete_scatter(X[:,0],X[:,1],y)
line = np.linspace(-15,15)
for coef, intercept, color in zip(linear_svm.coef_,linear_svm.intercept_,['b','r','g']):
    plt.plot(line,-(line*coef[0]+intercept)/coef[1],c=color)
plt.ylim(-10,15)
plt.xlim(-10,8)
plt.xlabel('Öznitelik 0')
plt.ylabel('Öznitelik 1')
plt.legend(['sınıf 0','sınıf 1','sınıf 2','doğru sınıf 0','doğru sınıf 1','doğru sınıf 2'],loc='best') # tutorialda loc=(1.02,0.4)