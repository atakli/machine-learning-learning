from sklearn.datasets import load_digits
__import__('matplotlib').interactive(True)
import matplotlib.pyplot as plt
digits = load_digits()
digits.images.shape
# 8x8 pixellik 1797 adet örneklem (sample)
fig, axes = plt.subplots(10,10,figsize=(8,8),subplot_kw={'xticks':[],'yticks':[]},gridspec_kw=dict(hspace=0.1,wspace=0.1))
# figsize dediği pencerenin boyutu
# 10,10 dediği 10 sütun 10 de satır olacak şekilde figürün içinde 100 tane figür olacak demek
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
