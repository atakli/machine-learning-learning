# coding: utf-8
# DVM sınıflandırma regresyon ve aykırı değerleri bulmak için kullanılan çok popüler bir supervised öğrenme tekniğidir
# el yazısı tanıma, zaman serisi analizi, konuşma tanıma gibi birçok alanda başarıyla uygulanmıştır
# bu derste lineer dvm, kernel dvm, hiperparametre ayarı gibi konulardan bahsedicez
### LİNEER DVM
# dvm'de herbir sınfı diğer sınıflardan ayırmak için doğru, iki boyutlu durumda eğri, çok boyutlu durumlarda manifold kullanıcaz
# şimdi iki sınıfın veri noktalarının güzel bi şekilde ayrıldığı make_blobs veri setini ele alalım
from sklearn.datasets import make_blobs
X,y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)
import matplotlib.pyplot as plt
__import__('matplotlib').interactive(True)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='autumn')
# bu iki topluluk arasında birçok doğru çizilebilir. yeni bir veri hangi doğru durumunda nereye dahil olacak? hangi doğrunun daha iyi ayırım yaptığını bilmiyoruz
# destek vektör makineleri sınıfları ayırmak için en iyi doğruyu bulmamızı sağlar
3 bunun için sınıfların en yakın nokralarına göre herbir doğrunun margin yani sınır çizgileri çiizlir. bu margini en büyük olan doğru modelin optimummu olarak seçilir
# bunun için sınıfların en yakın nokralarına göre herbir doğrunun margin yani sınır çizgileri çiizlir. bu margini en büyük olan doğru modelin optimummu olarak seçilir
from sklearn.svm import SVC
model = SVC(kernel='linear', c=1E10)
model = SVC(kernel='linear', c=1e10)
model = SVC(kernel='linear', C=1E10)
1e10 == 1E10
model.fit(X,y)
# margin meselesinin bu örneğpe tatbikine ait grafik 4.13'te
# margindeki (sınırdaki), en iyi doğruyu belirleten boktalara destek vektör ya da destek noktalar denir
# veri setini ayıtran en iyi doğru yalnız destek vektörüne göre çizilir
# diğer veri noktalarının önemi yoktur
# örneğin veri sayısı artsa ama destek vektörleri değişmese modelde herhangi bi değişiklik olmaz. bu durum dvm'nin gü
# çlü yönlerinden biridir
### KERNEL DVM
# lineer dvm birçok durumda iyi çalışmasına rağmen çoğu gerçek veri lineer bir doğruyla ayrılamaz
# dvmleri ayrıca kernels ile birleşince çok güçlü olurla
from sklearn.datasets.samples_generator import make_circles
X,y = make_circles(100, factor=.1, noise=.1)
clf = SVC(kernel='lineer').fit(X,y)
clf = SVC(kernel='linear').fit(X,y)
# bu veri setinin lineer bir doğruyla sınıflanamadığı 5.37'deki grafikte görülüyo
#bu veri setini ayırmak için bir öznitelik daha ekleyelim. böylece lineer bir doğru ile sınıflama yapabiliriz
#bir öznitelik eklemek için radial bases fonksiyonunu kullanalım
# eklenen öznitelik ile sınıflar bir düzlem kullanarak ayrılabilir hale geldş
# bu eklenecek özniteliği seçmek önemlidir. eğer doğur özniteliği seçmezsek sınıflamayı lineer şekilde yapamayız
# şanszlıyız ki burda kernel trick denen matematiksel bir kurnalık var. bu kurnazlık ile yükse boyutlu uzayda bir sınıflama öğrenilebilir
clf = SVC(kernel='rbf', C=1E6, gamma='auto').fit(X,y)
#görüldüğü gibi kernel'lı dvm kullanılarak  lineer olmayan karar sınırları öğrenildi
# kernel dönüşüm stratejisi ml'de lineer olmayan netotları lineer metotlara dönüştürmek için çok sık kullanılır
# biraz önce incelediğimiz veri seti sınıfların birbirinden iyi ayrıldığı temiz birveri setiydi. halbuki sınıfların arasındaji veri noktalrı birbirne çok yakın olabilir
# örneğin:
X,y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=1.2)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='autumn')
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='autumn')
# görüldüğü gibi bazı noktalar birbirne çok yakın. böyle durumlarda bazı nokraların margin içine kaymasına izin verilir
3 bu ayar c parametresiyle yapılır
# bu ayar c parametresiyle yapılır
# c büysükçe margin sıkıdır. verşş noktalrının margin içibe girmesi zorlaşr
# c küçük ise margin daha toleranslıdır
# dikkar!: margin dediğimiz iki kesikli çizgi arasındaki yer galiba. tam ortada da kesiksiz çiçzgi var
# c'nin optimum değeri veri setine ve cross validation ayarına bağlıdır
# veri noktaları arasındaki mesafe gaussian kernel ile ölçülür
# kesikçiz çizgi: karar sınırı
# dvm ile lineer olmayan bi çizgi çizilmiş. bu çizgi c ve gamma parametreleri ile düzenlenir
# unutma: c parametresi lineer modellerde olduğu gibi regülerleştirmeyi ve gamma parametresi gaussian kernel'ın genişliğini kontrol eder
# düşük gamma değeri: daha az kompleks model
# küçük c değeri modeli daha çok sınırlıyor. yani herbir veri noktası sınırlı etkiyee sahip. c büyüdükçe noktalar güçlü bir etkiye sahip
#ve karar sınıfları onları doğru bi şekilde sınıflamış
from sklearn.datasets import load_breast_cancer
kanser = load_breast_cancer()
from sklearn.model_selection import train_test_split
X_egitim, X_test, y_egitim, y_test = train_test_split(kanser.data, kanser.target, random_state=0)
svc = SVC(gamma='auto').fit(X_egitim,y_egitim)
svc.score(X_egitim,y_egitim)
svc.score(X_test,y_test)
# dvm parametre ayarlarına ve verinin ölçeklenmesine çok duyarlıdır
#bu teknilte bütün özniteliklerin benzer şekilde ölçeklenmesi gerekir
# şimdi öznitelikleri aynı ölçekle ölçekleyelim:
# bunun için eğitim setindeki her bir özelliğin min değerlerini bulalım:
min_on_training = X_test.min(axis=0)
# daha sonra herbir eğitim setindeki özniteliğin aralığını hesaplayalım
range_on_training = (X_test - min_on_training).max(axis=0)
X_train_scaled = (X_test - min_on_training) / range_on_training
min_on_training = X_train.min(axis=0)
min_on_training = X_egitim.min(axis=0)
range_on_training = (X_egitim - min_on_training).max(axis=0)
X_train_scaled = (X_egitim - min_on_training) / range_on_training
X_test_scaled = (X_test - min_on_training) / range_on_training
svc = SVC(gamma='auto').fit(X_train_scaled,y_egitim)
svc.score(X_egitim,y_egitim)
svc.score(X_train_scaled,y_egitim)
svc.score(X_test_scaled,y_test)
# şimdi de underfitting problemi çıktı
#bunun üstesinden helmek için c veya gammayı düzenle
# modelin biraz daha kompleks olması için c'yi artıralım
svc = SVC(gamma='auto', C=1000).fit(X_train_scaled,y_egitim)
print(svc.score(X_train_scaled,y_egitim))
print(svc.score(X_test_scaled,y_test))
# c artırılınca modelin anlamlılığı iyileşti
# özetle kernel'lı dvmleri çok güçlü modellerdir
# çeşitli veri setleri için iyi performans gösterir
# dvm'ler birkaç öznitelik olsa bile kompleks karar sınırlarına izin verir
# bu teknik hem düşük hem yüksek verilerde iyi çalışır
# model oluşturmada veri setinin büyüklüüğ önemli değildir
# yani onbin örneklemde iyi çalışan model yüzbin örneklemde de iyi çalışır
# diğer yandan parametrelerimn ayarına, preprocessing denen önişlemlerine dikkat etmek gerekir
# bazıları bu önişleme daha az gerektirdiği için dvm yerine  random forest gibi teknkleri kullanabiliyor
# gamma ve c parametreleri ike modelin kompleksliği kontrol edilir. bunların daha byük değerleri daha kmpleks modeller oluştutrur
bu oarametreler güçlü bi şeilde ilişkili olduğu için beraber düzenlenmelidir
# bu oarametreler güçlü bi şeilde ilişkili olduğu için beraber düzenlenmelidir
