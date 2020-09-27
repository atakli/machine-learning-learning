- from sklearn.neighbors import KNeighborsClassifier
- from sklearn.naive_bayes import GaussianNB
- from sklearn.linear_model import LinearRegression
- from sklearn.decomposition import PCA
- from sklearn.mixture import GaussianMixture
- from sklearn.manifold import Isomap
- from sklearn.feature_extraction import DictVectorizer
- from sklearn.feature_extraction.text import CountVectorizer
- from sklearn.preprocessing import PolynomialFeatures
- from sklearn.impute import SimpleImputer
- from sklearn.pipeline import make_pipeline # sadece satır azaltmak için gibi duruyo


- iris:
# GaussianNB
model.score(X_egitim,y_egitim): 0.9464285714285714
model.score(X_test,y_test): 0.9736842105263158
# KNeighborsClassifier (neighbors adedi 1 iken. artırdıkça bazen sabit kalıyo, bazen azalıyo. ama azalışı çok yavaş)
knn.score(X_eğitim,y_eğitim): 1.0
knn.score(X_test,y_test): 0.9736842105263158 # laan. üsttekiyle nasıl aynı çıktı bu meret?
# PCA. unsupervised learning. feature azaltma (n_components=2): Isomap'i denedim. değişik görünüyo. ama iyi gibi, not sure
# GaussianMixture. unsupervised learning. targetları bilmediği halde bu kadar başarılı olması şaşırttı. acaba yanlış mı yaptım:
0.9666666666666667

- rastgele doğrusalımsı bir veri:
# LinearRegression
model.score(X,y): 0.9638084659824165
- rastele kategorik ve stringten oluşan bir veri:
# DictVectorizer 
bu da heralde unsupervised. çünkü kategorik string'i integer'a çevirmek için kullanıldı
# CountVectorizer
bu da benzer. en son pd.DataFrame ile güzel hale geldi. ama bi eksiği var gibi: kelimelerin sırasını gözetmiyo
- rastgele bir veri:
# PolynomialFeatures
LinearRegression yetmedi çünkü veri lineer değil. 3. dereceden PolynomialFeatures kullandık. hatta 4 olunca tam oldu.
Öznitelik türetmiş olduk.
- rastgele ve nan veriler içeren veri:
# SimpleImputer
heralde unsupervised. eksik verileri tamamlıyo


- digits:
# GaussianNB
model.score(X_egitim,y_egitim) 0.8574610244988864
model.score(X_test,y_test) 0.8333333333333334
# Isomap: unsupervised learning. feature azaltma (n_components=2): PCA'yı denedim. isomap daha başarılı bunda. neden onu seçtik?

- mglearn.datasets.make_forge:
# KNeighborsClassifier

- load_breast_cancer:
