from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups # bu gereksiz 
kategoriler = ['talk.religion.misc','soc.religion.christian','sci.space','comp.graphics']
train = fetch_20newsgroups(data_home=r'C:\Users\Emre\AppData\Local\Programs\Python\Python38-32\Lib\site-packages\sklearn\datasets\data',subset='train',categories=kategoriler)
test = fetch_20newsgroups(data_home=r'C:\Users\Emre\AppData\Local\Programs\Python\Python38-32\Lib\site-packages\sklearn\datasets\data',subset='test',categories=kategoriler)
# klasör ismi yazmadığımda da allahu alem o klasöre çıkaracakmış. beklemem gerekiyomuş. beklesem heralde olacakmış. asıl uzun süren indirme değilmiş. archive dosyasını açmakmış. arşivi elle açarken de çok uzun sürdü. ama nerden bilebilirdim ki? scriti yazan herif adam gibi açıklayıcı notlar yazsaydı sorun olmazdı. may take a few minutes deyişini klasik zannettim. indiriliyor, indirme tamamlandı, şimdi arşivden çıkarılıyor, şu klasöre çıkarılıyor, arşivden çıkarma uzun sürebilir vs yazmalıydı. ben ekleyebiilirim scripte
# C:\Users\Emre\AppData\Local\Programs\Python\Python38-32\Lib\site-packages\sklearn\datasets\_twenty_newsgroups.py
# ahanda buraya
print(train.data[5])
# bu veri seti sınıflandırma ve kümeleme gibi makine öğrenmesi tekniklerinin test uygulamalarında kullanılan popüler bir evridir
# 20 farklı konuda haber yazıları içeriyor
# bu veri setini makin eöğrenesinde kullanmak için herbir stringin içeriğini bir sayısal vektöre çevirmemiz gerekiyor
from sklearn.feature_extraction.text import TfidfVectorizer
# çok kategorili naive bayes sınıfını imğport edelim:
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
model = make_pipeline(TfidfVectorizer(),MultinomialNB())
model.fit(train.data,train.target)
etiketler = model.predict(test.data)
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(test.target,etiketler)
import seaborn as sns
__import__('matplotlib').interactive(True)
import matplotlib.pyplot as plt
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=train.target_names,yticklabels=train.target_names)
plt.xlabel('Gerçek değerler')
plt.ylabel('Tahmin etiketleri')
# herhangi bir stringin kategorilerini belirlemek için bir toola sahibiz
def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]
predict_category('discussing islam vs atheism')
predict_category('determining the screen resolution')
### NE ZAMAN KULLANILIR?
# NAİVE BAYES SINIFLANDIRMASINDA veri hakkında katı varsayımlar olduğundan dolayı genellikle karışık modeller için kullanılmaz
# bu sınıflandırmanın avantajları: hem eğitimde, hem de tahmin yapmada çok hızlıdır
# yorumlanması kolaydır
# çok az ayar parametresine sahiptir
# başlangıç için iyi bir seçimdir
# bu iyi çalışırsa şanslısın. çünkü problem hızlı ve kolayca çözülür
# iyi çalışmazsa daha karışık modeller denenir
# peki bu model hangi durumlarda iyi performans gösterir?
# öncelikle naive varsayımları veriyle uyuşuyorsa model kompleksliği daha az önemli olduğu zaman kategoriler iyi bi şekilde ayrılmışsa
# ve yüksek boyutlu veriler varsa naive bayes modeli iyi çalışır