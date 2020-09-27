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
# bu iki topluluk arasında birçok doğru çizilebilir. yeni bir veri hangi doğru durumunda nereye dahil olacak?
# hangi doğrunun daha iyi ayırım yaptığını bilmiyoruz
# destek vektör makineleri sınıfları ayırmak için en iyi doğruyu bulmamızı sağlar
# bunun için sınıfların en yakın noktalarına göre herbir doğrunun margin yani sınır çizgileri çiizlir. 
# bu margini en büyük olan doğru modelin optimumu olarak seçilir
from sklearn.svm import SVC
model = SVC(kernel='linear', C=1E10)
model.fit(X,y)
# margin meselesinin bu örneğe tatbikine ait grafik 4.13'te
# margindeki (sınırdaki), en iyi doğruyu belirten noktalara destek vektör ya da destek noktalar denir
# veri setini ayıran en iyi doğru yalnız destek vektörüne göre çizilir, diğer veri noktalarının önemi yoktur
# örneğin veri sayısı artsa ama destek vektörleri değişmese modelde herhangi bi değişiklik olmaz.
# bu durum dvm'nin güçlü yönlerinden biridir
### KERNEL DVM
# lineer dvm birçok durumda iyi çalışmasına rağmen çoğu gerçek veri lineer bir doğruyla ayrılamaz
# dvm'ler ayrıca kernels ile birleşince çok güçlü olur
from sklearn.datasets.samples_generator import make_circles
X,y = make_circles(100, factor=.1, noise=.1)
clf = SVC(kernel='linear').fit(X,y)
# bu veri setinin lineer bir doğruyla sınıflanamadığı 5.37'deki grafikte görülüyo
# bu veri setini ayırmak için bir öznitelik daha ekleyelim. böylece lineer bir doğru ile sınıflama yapabiliriz
# bir öznitelik eklemek için radial bases fonksiyonunu kullanalım
# eklenen öznitelik ile sınıflar bir düzlem kullanarak ayrılabilir hale geldi
# bu eklenecek özniteliği seçmek önemlidir. eğer doğru özniteliği seçmezsek sınıflamayı lineer şekilde yapamayız
# şanslıyız ki burda kernel trick denen matematiksel bir kurnazlık var. 
# bu kurnazlık ile yüksek boyutlu uzayda bir sınıflama öğrenilebilir
clf = SVC(kernel='rbf', C=1E6, gamma='auto').fit(X,y)
# görüldüğü gibi kernel'lı dvm kullanılarak lineer olmayan karar sınırları öğrenildi
# kernel dönüşüm stratejisi ml'de lineer olmayan metotları lineer metotlara dönüştürmek için çok sık kullanılır
# biraz önce incelediğimiz veri seti sınıfların birbirinden iyi ayrıldığı temiz bir veri setiydi.
# halbuki sınıfların arasındaki veri noktaları birbirine çok yakın olabilir. örneğin:
X,y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=1.2)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='autumn')
# görüldüğü gibi bazı noktalar birbirine çok yakın. böyle durumlarda bazı noktaların margin içine kaymasına izin verilir
# bu ayar c parametresiyle yapılır
# c büyükse margin sıkıdır. vergi noktalarının margin içine girmesi zorlaşır
# c küçük ise margin daha toleranslıdır
# dikkar!: margin dediğimiz iki kesikli çizgi arasındaki yer galiba. tam ortada da kesiksiz çizgi var
# c'nin optimum değeri veri setine ve cross validation ayarına bağlıdır (? o ne)
# veri noktaları arasındaki mesafe gaussian kernel ile ölçülür
# kesiksiz çizgi: karar sınırı
# dvm ile lineer olmayan bi çizgi çizilmiş. bu çizgi c ve gamma parametreleri ile düzenlenir
# unutma: c parametresi lineer modellerde olduğu gibi regülerleştirmeyi
# ve gamma parametresi gaussian kernel'ın genişliğini kontrol eder
# düşük gamma değeri: daha az kompleks model
# küçük c değeri modeli daha çok sınırlıyor. yani herbir veri noktası sınırlı etkiye sahip.
# c büyüdükçe noktalar güçlü bir etkiye sahip
# ve karar sınıfları onları doğru bi şekilde sınıflamış
from sklearn.datasets import load_breast_cancer
kanser = load_breast_cancer()
from sklearn.model_selection import train_test_split
X_egitim, X_test, y_egitim, y_test = train_test_split(kanser.data, kanser.target, random_state=0)
svc = SVC(gamma='auto').fit(X_egitim,y_egitim)
print(svc.score(X_egitim,y_egitim))
print(svc.score(X_test,y_test))
# dvm parametre ayarlarına ve verinin ölçeklenmesine çok duyarlıdır
# bu teknikte bütün özniteliklerin benzer şekilde ölçeklenmesi gerekir
# şimdi öznitelikleri aynı ölçekle ölçekleyelim:
# bunun için eğitim setindeki her bir özelliğin minimum değerlerini bulalım:
min_on_training = X_egitim.min(axis=0)
# daha sonra herbir eğitim setindeki özniteliğin aralığını hesaplayalım
range_on_training = (X_egitim - min_on_training).max(axis=0)
X_train_scaled = (X_egitim - min_on_training) / range_on_training
X_test_scaled = (X_test - min_on_training) / range_on_training
svc = SVC(gamma='auto').fit(X_train_scaled,y_egitim)
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
