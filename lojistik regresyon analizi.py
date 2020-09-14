1# coding: utf-8
### VERİYİ TANIMAKNIN EN GÜZEL YOLU VERİYİ GÖRESELLEŞTRMEKTİR
import pandas as pd
pd.read_csv('titanic.csv') # Titanik veri seti için link : bit.ly/2MHdIqu
veri = pd.read_csv(r'C:\Users\Emre\Downloads\nrippner-titanic-disaster-dataset\original\titanic.csv') 
# Titanik veri seti için link : bit.ly/2MHdIqu
veri.head()
veri.shape
veri.dtypes
import seaborn as sns
sns.countplot(x='survived',data=veri)
sns.countplot(x='survived',data=veri,hue='pclass')
sns.countplot(x='survived',data=veri,hue='sex')
from mplcursors import cursor
cursor(multiple=True)
veri['age'].plot.hist()
cursor(multiple=True)
# bu histogramı tam anlamadım. 1 yaşında ile  2 yaşındaki çocuk sayısı aynı mı?
veri['fare'].plot.hist(bins=20,figsize=(10,5))
sns.countplot(x='sibsp',data=veri)
### BÖYLECE VERİYİ ÖNİNCELEDİK ŞİMDİ VERİ ÖNİŞLEME YAPALIM
# eksik yada kayıp denilen missing data veri analizinde başa beladır. çünkü eksik veri bizi yanlış yönlendirir
# öncelikle veri setinde eksik veri var mı kontrol edelim
veri.isnull()
# True'lar eksik veri olduğunu gösteriyo. 
import numpy as np
veri.isnull().sum() # veya np.sum(_)
import seaborn as sns
sns.heatmap(veri.isnull(),yticklabels=False,cmap="viridis")
# eksik verisi fazla olan sütunları kaldıralım
veri.drop(["cabin","body","boat","home.dest"],axis=1,inplace=True)
veri.head()
sns.heatmap(veri.isnull(),yticklabels=False,cmap="viridis")
veri.isnull().sum()
# yine de eksik veriler var. bu eksik verileri çeşitli yöntemlerle doldurabiliriz
# bu analizde eksik veri içeren satırları kaldıralım
# eksik verilerin nasıl doldurulacağını öğrenmek için tirendaz'ın pandas eksik veriler dersini izle
veri.dropna(inplace=True)
veri.isnull().sum()
veri.dtypes
# object tipindeki değişkenlerin analize dahil edilebilmesi için dummy tipine dönüştürmek gerek
sex = pd.get_dummies(veri["sex"],drop_first=True) # ilk alt seviyeyi kaldırmak için drop_first True yapıldı
sex.head()
# embarked değişkeninin alt gruplarının sayısını bulalım:
veri.embarked.value_counts()
# şimdi embarked ve pclass'ı dummy yapalım
embarked = pd.get_dummies(veri["embarked"],drop_first=True)
pclass = pd.get_dummies(veri["pclass"],drop_first=True)
veri.drop(["sex","embarked","pclass"],axes=1,inplace=True)
veri = pd.concat([veri,sex,pclass,embarked],axis=1)
veri.dtypes
# tutorial'da veri.dtypes farklı sonuç verdi
# concat ile eklenen değişkenler tutorial'da yok bende var. bende var ama değişmemiş, hala object türünde?
# ya anlamadım: madem embarked ve diğer ikisini drop edecektik, neden type'larını değiştirdik
# zaten kullanmayacağımız name ile ticket'ı kaldıralım:
veri.drop(["name","ticket"],axis=1,inplace=True)
veri.head()
# böylece veri önişlemlerini bitirdik. 
# veri bilimcilerinin en çok zorlandıkları kısım veri temizleme ve veri önişleme işlemleridir
# artık esas analize geçebiliriz
### LOJİSTİK REGRESYON ANALİZİ
veri.drop(["sex","pclass","embarked"],axis=1,inplace=True) # dedim madem tutorial'da bunlar yok, kaldırayım
veri.head()
# yapacağımız analizde amacımız verilerini girdiğimiz kişinin hayatta kalıp kalmamasını tahmin etmek
# yani hedef değişken: survived
# buna bazen bağımlı değişken veya sonuç değişkeni de denir. diğer değişkenlere öznitelik veya bağımsız değişken denir
X = veri.drop(["survived"],axis=1) # analiz etmek için girdi ve sonuç verilerini oluşturuyoruz
y = veri["survived"]
from sklearn.model_selection import train_test_split
X_egitim, X_test, y_egitim, y_test = train_test_split(X,y,random_state=100,test_size=0.25)
# test_size veri setinin hangi oranda bölüneceğini belirler. help'e baktım, default zaten 0.25 gibi gördüm. neden yazdık o zaman?
# hedef değişken kategorik olduğu için lojistik regresyon analizini kullanıcaz
from sklearn.linear_model import LogisticRegression
lg_model = LogisticRegression()
lg_model.fit(X_egitim,y_egitim)
# max_iter 100 iken uyarı verdi, artırmamız istendi:
lg_model = LogisticRegression(max_iter=1000)
lg_model.fit(X_egitim,y_egitim)
lg_model.score(X_egitim,y_egitim)
lg_model.score(X_test,y_test)
# train_test_split lafta random ayırıyo ama bendekiyle tutorial'daki aynı :) 
# iradesi olmadığı, illaki bi algoritmaya göre çalıştığı için tam random olamaz allahu alem
# bunun anlamı: kurduğumuz model yeni bir veriyi %77 oranında doğru tahmin edicek
# unutmayın, modelin eğitim ve test verisindeki doğruluk skorlarının birbirrne yakın olmasını isteriz
# modelimiz eğitim verisinde biraz daha iyi. bu durum modelde biraz overfitting probleminin olduğunu gösteriyor
# bu problemin üstesinden gelmek için modeli regülerleştirelim. Bunun için C argümanını kulllanıcaz
lg_model = LogisticRegression(max_iter=1000,C=0.1)
# C'nin optimum değerini bulacak br metod yok mu? grafik çizerek de görülebilir ama direk metod olmalı
lg_model.fit(X_egitim,y_egitim)
lg_model.score(X_test,y_test)
lg_model.score(X_egitim,y_egitim)
# değerler birbirine yaklaştı. 1'e çok yakın değiller ama yine de fena değil
### MODEL DEĞERLENDİRME
from sklearn.metrics import confusion_matrix
tahmin = lg_model.predict(X_test)
confusion_matrix(y_test,tahmin)
# 124 tane true positif, 80 tane true negatif tahmin yapılmış. 21 tane False negatif, yani positif olmasına rağmen negatif denmiş
# şimdi denemek için ilk kişinin verilerine bakalım
yeni_veri = np.array([[29,0,0,211.3375,0,0,0,0,1]])
# şimdi farkettim: bende tutorial'dan faklı bi sonuç çıkmıştı ya, onun sebebi şu olabilir: bende sütunların yerleri biraz farklı
# array'i böyle veri.head'e bakarak yazmak çok amelece. python'ın ruhuna muhalif. şöyle yapılmalı mesela:
yeni_veri_mine = np.array(X.loc[0]).reshape(1,X.shape[1]) 
# bu da çok kısa değil aslında :) ama olsun. mesela feature sayısı çok da olabilirdi. hem bakarak yazmak hataya daha çok kabil
lg_model.predict(yeni_veri) # doğru tahmin etti
lg_model.coef_ 
yeni_veri2 = np.array([[30,1,1,150,0,0,1,0,0]]) # rastgele veri
lg_model.predict(yeni_veri2)

