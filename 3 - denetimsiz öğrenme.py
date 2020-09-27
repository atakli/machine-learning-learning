### FEATURE AZALTMA 
# denetimsiz (unsupervised) öğrenme problemi olarak iris veri setinin boyutunu (öznitelik) azaltma
__import__('matplotlib').interactive(True)
import matplotlib.pyplot as plt
import seaborn as sns
iris = sns.load_dataset('iris')
iris.head()
X_iris = iris.drop('species', axis=1)
y_iris = iris['species']

from sklearn.decomposition import PCA
model = PCA(n_components=2)
model.fit(X_iris)
X_2D = model.transform(X_iris)
iris['PCA1'] = X_2D[:,0]
iris['PCA2'] = X_2D[:,1]
iris.head()
sns.lmplot('PCA1','PCA2',hue='species',data=iris,fit_reg=True)
# PCA algoritması tüm etiketleri bilmemesine rağmen türleri oldukça iyi ayırdı
# bu grafik veri seti için basit sınıflandırma yönteminin kullanılabileceğini gösteriyo

### KÜMELEME ANALİZİ
# bu da denetimsiz öğrenme algoritması
# bu herhangi bir etiket referansı olmayan grupları ayırmak için kullanılır
# güçlü bir kümeleme modeli olan GaussianMixture'ı kullanıcaz
from sklearn.mixture import GaussianMixture
model = GaussianMixture(n_components=3, covariance_type='full')
model.fit(X_iris) # dikkat ettiysen, denetimsiz öğrenme modeli olduğu için y_hedef değişkenini burada kullanmadık
y_gmm = model.predict(X_iris)
iris['kumeleme'] = y_gmm
iris.head()
sns.lmplot('PCA1','PCA2',hue='species',data=iris, col='kumeleme',fit_reg=True)
# bu plot da var ama ben yüzde olarak görmek istedim. ben bişey yaptım ama daha kolay yolu olmalı: 0.9666666666666667


### DENETİMSİZ ÖĞRENME İLE İLGİLİ UYGULAMA

from sklearn.datasets import load_digits
digits = load_digits()
digits.images.shape
# 8x8 pixellik 1797 adet örneklem (sample)
fig, axes = plt.subplots(10,10,figsize=(8,8),subplot_kw={'xticks':[],'yticks':[]},gridspec_kw=dict(hspace=0.1,wspace=0.1))
# figsize dediği pencerenin boyutu
# 10,10 dediği 10 sütun 10 da satır olacak şekilde figürün içinde 100 tane figür olacak demek
# subplot_kw'yi kaldırınca figürün heryerinde karman çorman sayılar filan çıktı ve axesler birbirine girdi
for i,ax in enumerate(axes.flat):
    ax.imshow(digits.images[i],cmap='binary',interpolation='nearest')
    ax.text(0.05,0.05,str(digits.target[i]),transform=ax.transAxes,color='green')
# interpolation argümanını kaldırdım bişey değişmedi
X = digits.data
y = digits.target
# X dediğimiz öznitelik (feature) dizisi
X.shape # (1797, 64) digits.images'ın flattened hali
y.shape
# şimdi öznitelik matrisindeki 64 boyut yani özniteliği görselleştirmek isteyelim
# fakat böyle yüksek boyutlu veriyi görselleştirmek zordur. bunun için denetimsiz öğrenme metotlarını kullanarak
# veriyi iki boyuta düşürelim
from sklearn.manifold import Isomap
iso = Isomap(n_components=2)
iso.fit(X)
data2 = iso.transform(X)
data2.shape
# veri iki boyutlu hale geldi. mükemmel
# şimdi veriyi tanımak için grafiğini çizdirelim
plt.figure()
plt.scatter(data2[:,0],data2[:,1], c=digits.target, alpha=0.5, cmap=plt.cm.get_cmap('tab10',10))
plt.colorbar(label='digit etiket', ticks=range(10))
# bu grafikten 64 boyut durumunda rakamların birbirinden ne kadar iyi ayrılacağını sezgisel olarak anlıyoruz
from sklearn.model_selection import train_test_split
X_egitim, X_test, y_egitim, y_test = train_test_split(X,y,random_state=0)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_egitim,y_egitim)
y_model = model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_model,y_test)
# model.score(X_egitim,y_egitim) 0.8574610244988864
# model.score(X_test,y_test) 0.8333333333333334
# aşırı basit bir model ile bile rakamları sınıflandırma modelinin doğruluk oranı bu çıktı
# fakat bu değer bizim nerede yanlış yaptığımızı söylemiyor
# confusion matrix kullanarak yanlışlıkları bulabiliriz
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test,y_model)
import seaborn as sns
plt.figure()
sns.heatmap(mat,square=True,annot=True,cbar=True)
plt.xlabel('Tahmin Değer')
plt.ylabel('Gerçek Değer')
# annot'u kaldırınca karelerin içindeki sayılar gitti. cbar zaten kenardaki renk skalası
# square'i kaldırınca enlemesine genişledi, heralde square'ler kare olmaktan çıktı
