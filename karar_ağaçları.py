# karar ağaçları dvm gibi hem sınıflandırma hem de regresyon için yaygın olaarak kullanılan modellerdendir
# bunlar karar almak için if else sorularının bir hiyerarşisini öğrenirler
# yani bir dizi sorular sorarsınız, cevaba göre karar alırsınız
# karar ağaçları hem de random forest olarak adlandırılan numparametrik (?) algoritmaların temel bileşenlerinden biridir
# bu derste şunlar var: sınıflandırma için karar ağaçları, görselleştirme, regresyon için karar ağaçları, avantajlar
# ağaçlardaki herbir düğüm veya yaprak bir soruyu gösterir
# ml mantığı ile kuş tüyü var mı, uçabiliyor mu, yüzgeci var mı gibi üç özniteliği kullanarak 
# belirlediğimiz dört hayvan grubunu ayırdık: şahin, devekuşu, yunus, ayı
# karar ağaçları bütün yapraklar poor (?) yani yalın olana kadar devam ederse model kompleks olur. 
# bu durumda overfitting problemi ortaya çıkar
# overfittingi engellemek için iki genel strateji vardır: ağaç tamamlanmadan dallanmayı durdurmak, buna ilk budama denir. 
# diğer yöntem: ağaç tamamlanır ama az bilgi içeren yapraklar kaldırılır. buna da son budama denir
# sklearn'de decision tree regressor ve classifier var. sklearn sadece ilk budamayı uygular
### SINIFLANDIRMA İÇİN KARAR AĞAÇLARI
from sklearn.datasets import load_breast_cancer
kanser = load_breast_cancer()
from sklearn.model_selection import train_test_split
X_egitim, X_test, y_egitim, y_test = train_test_split(kanser.data, kanser.target, stratify=kanser.target)
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(X_egitim,y_egitim)
print(tree.score(X_egitim,y_egitim))
print(tree.score(X_test,y_test))
# beklediğimiz gibi eğitim skoru %100 çıktı
__import__('matplotlib').interactive(True)
from sklearn.datasets import load_breast_cancer
kanser = load_breast_cancer()
from sklearn.model_selection import train_test_split
X_egitim, X_test, y_egitim, y_test = train_test_split(kanser.data, kanser.target, stratify=kanser.target)
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(X_egitim,y_egitim)
print(tree.score(X_egitim,y_egitim))
print(tree.score(X_test,y_test))
# çümnkü yapraklar en yalın duırumda. (o ne demek ya)
# ağaç inebileceği en derine indi ve eğitim verisindeki büyün etiketleri mükemmel şekiğlde etiketledi
# eğer karar ağacının derinliğini sınırlamazsak ağaç çok derin ve kompleks olur.
# bu durumda modelin genelleştirilmesinde problem ortaya çıkar
# şimdi ön budama yapalım. yani model eğitim verisine mükemmel fit edilmeden önce ağacın gelişimini durduralım
tree = DecisionTreeClassifier(max_depth=4) # böylece eğitm skoru düşecek, test skoru artacak 
tree.fit(X_egitim,y_egitim)
print(tree.score(X_egitim,y_egitim))
print(tree.score(X_test,y_test))
# böylece modlein performansını artırmış olduk
### GÖRSELLEŞTİRME
# karar ağaçlarını daha iyi anlamak için görselleştirelim
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data[:,2:]
y = iris.target
# ağaç derinliği iki olacak şekilde decision tree classifier sınıfından bir örnek alalım
tree = DecisionTreeClassifier(max_depth=2)
tree.fit(X,y)
from sklearn.tree import export_graphviz
export_graphviz(tree,out_file='tree.dot',feature_names=iris.feature_names[2:],class_names=True,filled=True)
# karar ağacı diagramını .dot formatında dosyaya yazdırmış olduk. oluşan dosyayı dizinde görebiliriz. uygun bir programla açabilir
# veya png, pdf gibi formatlara dönüştürebiliriz
# graphviz paketini import ederek de karar ağacını görebiliriz
import graphviz
with open('tree.dot') as f: # tutorial'da with olmadan çalıştırdı, nasıl bilmiyorum
    dot_graph = f.read()
graphviz.Source(dot_graph) # Source kelimesini sörç diye telaffuz etti. bu adamdan ders alıyorum
# grafik çıkmadı
import matplotlib.pyplot as plt
plt.show(block=False)
# tutorial'da görselleştirmiş olduk diyor. hani nerde?
# karar ağaçlarında veriyi ölçeklendirmek gibi nadiren veriyi önişlemek gerekir
# görselde (which I could not create here) value dediği "sınıf başına düşen örneklem"i gösteriyormuş
# (value'ların toplamı samples ediyor)
# ağacın deirnlerine indikçe bu örneklemler ve valuelar bölünerek devam eder
# düğümlerdeki "gini" attribute'u sınıfın yalınlığını gösterir
# eğer gini 0 ise bu sınıf tamamen yalın, yani eğitim örneklerinin tamamı aynı sınıfa aittir demektir
### REGRESYON İÇİN KARAR AĞAÇLARI
from sklearn.tree import DecisionTreeRegressor
# burda value dediği bu düğüm ile ilişkili yüzon eğitim örneklemin hedef deperinin ortalamasıdır
# bu tahminin ortalama kareler hatası yaklaşık 0.02'dir (mse'yi diyo heralde) (12.42)
# regresyonda overfittinh problemi ile karşılaşma ihitmali fazladır
# karar ağaçları kolayca görselleştirildiği için yotumlaması kolayudır ve veri ölçeklendirme gini veri önişlemeye çok az ihtiyaç duyar