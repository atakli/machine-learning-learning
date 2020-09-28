# öznitelik sayısı arttıkça modelin kompleksliği artar ama bu sefer model ezberlemeye başlar
# ezberleme overfitting problemini ortaya çıkarır. eğer model ezberlerse modeli genelleştiremeyiz, yeni bir veriyi iyi tahmin edemez
# unutma, ezberleyen modelin eğitim skoru yüksek ama test skoru düşüktür
# ezberleme probleminin üstesinden gelmek için ridge ve lasso gibi teknikler kullanılır
# ridge regresyon da linear bir modeldir
# ridge'de bir kısıtlama ile fit edilir. katsayıların değeri mümkün olduğunca küçük olması, hatta 0'a yakın olması istenir
# bunun anlamı: bir yandan iyi bir tahmin yapmak istenirken, 
# diğer yandan herbir özniteliğin mümkün olduğunca sonuca az etki etmesi sağlanır
# bu kısıtlamaya regülerleştirme denir
# regülerleştirme ile overfittingden kaçınmak için model kısıtlanır
# bu vidyoda bazı tabirleri anlamadım, büzülme falan. geçiyorum, sonra tekrar izlenebilir (la oğlum hangi dk'da olduğunu yazsaydın ya)
from sklearn.datasets import load_boston
boston = load_boston()
print(boston.DESCR)
# veya print(boston['DESCR'])
import mglearn
X,y = mglearn.datasets.load_extended_boston()
# böyle yapacağına şu da olur: X = boston.data y = boston.target. Ama boston.data'nın shape'i farklı. 
# heralde columnların bir kısmını silersek olur. edit: hangisi kimbilir? zor iş.
# ama y ile boston.target aynı
# 506 örneklem sayısını, 104 de 13 tane öznitelik sayısını ve bu özniteliklerin ikişerli ilişkilerini gösteriyor.
from sklearn.model_selection import train_test_split
X_egitim, X_test, y_egitim, y_test = train_test_split(X,y,random_state=0)
from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_egitim,y_egitim)
print(ridge.score(X_egitim,y_egitim))	# 0.8857966585170939
print(ridge.score(X_test,y_test))		# 0.7527683481744751
# aslında ridge modelin basitliği ile eğitim verisindeki modelin performansı arasında trade-off'u sağlar
# modelin eğitim verisindeki performansına karşı basitliğin ne kadar olacağını alpha parametresi belirler
# bu alpha öntanım olarak 1 gelir
# alpha arttıkça katsayılar 0'a doğru daha çok yaklaşır
lr = LinearRegression().fit(X_egitim,y_egitim)
print(lr.score(X_egitim,y_egitim))	# 0.9520519609032728
print(lr.score(X_test,y_test))		# 0.6074721959665891
# bu durum modelin eğitim verisindeki performansını azaltırken modelin genelleştirilmesine yardım edebilir
ridge10 = Ridge(alpha=10).fit(X_egitim,y_egitim)
print(ridge10.score(X_egitim,y_egitim))	# 0.7882787115369614
print(ridge10.score(X_test,y_test))		# 0.635941148917731
# alpha'nın değeri azaldıkça modele girecek katsayılar daha az sınırlandırılır
ridge01 = Ridge(alpha=0.1).fit(X_egitim,y_egitim)
print(ridge01.score(X_egitim,y_egitim))	# 0.9282273685001993
print(ridge01.score(X_test,y_test))		# 0.7722067936479815
# bu sefer katsayılar daha az sınırlandırıldığı için model lineer regresyona benzemiş oldu
# alpha'nın 0.1 değeri için test ve eğitim verilerinin doğruluk skorları yükseldi. dolayısıyla model daha iyi çalıştı
### LASSO
# lasso da bir lineer regresyon modelidir
# lineer regresyona mutlak değer kullanılarak eklenen regülerleştirme teriminden dolayı lassoya ...'li regülerleştirmesi de denir
# lasso öznitelik sayısını önemli ölçüde azaltır
# lasso da ridge gibi katsayıları sıfıra yaklaştırmak için kısıtlar
# lasso regülerleştirmesi ile bazı katsayılar 0 alınır
# yani veri setindeki bazı öznitelikler modele dahil edilmez
# böylece modelde daha önemli öznitelikler yer alır
# bazı katsayılar 0 olunca model sadeleşir ve daha kolay yorumlanır
# edit: sanki vidyoda vardı ama emin değilim: galiba ridge'de 0'a yaklaşıyo, lasso'da 0 oluyo
from sklearn.linear_model import Lasso
lasso = Lasso().fit(X_egitim,y_egitim)
print(lasso.score(X_egitim,y_egitim))	# 0.29323768991114596
print(lasso.score(X_test,y_test))		# 0.20937503255272272
# sonuçlar oldukça kötü. yani overfitting var. yani modelde değişken sayısı az (öyle yazmışım ama underfitting olmalı sanki)
# edit: evet. underfitting olmalı. 6. tutorial dosyasındaki şu not da öyle olması gerektiğini teyit ediyo:
# "model basit olduğundan underfitting var. bu, sonuç değişkenini açıklamak için başka değişkenlere ihtiyaç var anlamına gelir"
# modelde kullanılan değişken sayısını görmek isteyelim:
import numpy as np
np.sum(lasso.coef_!=0) 
# lasso.coef_!=0 True False dizisi veriyo. onları topluyo. True 1 demek
# demekki 104 tane öznitelikten modelde sadece 4'ü kullanılmış. edit: fesubhanallah
# ridge gibi lasso'da da regülerleştirme parametresi alpha kullanılır
# alpha değeri katsayıları sıfıra çekmek için modeli ne kadar zorlayacağımızı gösterir
# underfitting'i azaltmak için alpha'yı azaltalım
# bunun için max_iter'i artırmamız gerekir
lasso001 = Lasso(alpha=0.01,max_iter=100000).fit(X_egitim,y_egitim)
print(lasso001.score(X_egitim,y_egitim))# 0.8962226511086496
print(lasso001.score(X_test,y_test))	# 0.7656571174549979
# düşük alpha daha kompleks modelin kurulmasını sağladı
# böylece model eğitim ve test verieri için daha iyi çalıştı
# bu modelin performansı ridge ile kurulan modelden daha iyi çıktı. edit: yoo? 0.93 & 0.77 bulmuştuk ya?
# modelde 104 öznitelikten yalnız 33'ü kullanıldı
# bu modelin daha iyi anlaşılmasını sağlar
# alpha'yı biraz daha azaltabiliriz ama bu sefer de regülerleştirmenin etkisi azalır
# böylece lineer regresyona yaklaşır:
lasso00001 = Lasso(alpha=0.00001,max_iter=100000).fit(X_egitim,y_egitim) #dedi ki: You might want to increase the number of iterations
print(lasso00001.score(X_egitim,y_egitim))	# 0.9515087977585444
print(lasso00001.score(X_test,y_test))		# 0.6193582257352808
# pratikte bu iki modelden öncelikle ridge regresyonu tercih edilir
# fakat öznitelik sayısı fazla ve bunlardan bazılarının daha önemli olduğu düşünülüyorsa lasso daha iyi bir tercihtir
### ELASTICNET
# ridge ve lasso'nun birleşimi
# lasso gibi katsayılardan bazılarını sıfır yaparken ridge'in regülerleştirme özelliğini koruyor. edit: neyyy?
# pratikte bu birleşimin daha iyi çalıştığını söyleyebiliriz
# elsaticnet katsayıların bağlantılı olduğu durumlarda da kullanılabilir
# lineer regresyonda sonuç değişkeni sürekli sayısal değişkendi. 
# sonuç değişkeni kategorik ise analiz için lojistik regresyon kullanılabilir