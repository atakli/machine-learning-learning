- from sklearn.neighbors import KNeighborsClassifier
- from sklearn.neighbors import KNeighborsRegressor
- from sklearn.naive_bayes import GaussianNB
- from sklearn.linear_model import LinearRegression
- from sklearn.linear_model import Ridge
- from sklearn.linear_model import Lasso

- from sklearn.impute import SimpleImputer
- from sklearn.mixture import GaussianMixture
- from sklearn.manifold import Isomap
- from sklearn.decomposition import PCA
- from sklearn.preprocessing import PolynomialFeatures
- from sklearn.feature_extraction import DictVectorizer
- from sklearn.feature_extraction.text import CountVectorizer

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
Öznitelik türetmiş olduk. unsupervised olabilir, ama not sure. hiçbişey de olmayabilir
- rastgele ve nan veriler içeren veri:
# SimpleImputer
heralde unsupervised. eksik verileri tamamlıyo


- digits:
# GaussianNB
model.score(X_egitim,y_egitim) 0.8574610244988864
model.score(X_test,y_test) 0.8333333333333334
# Isomap: unsupervised learning. feature azaltma (n_components=2): PCA'yı denedim ama isomap daha başarılı bunda. neden onu seçtik?

- mglearn.datasets.make_forge:
# KNeighborsClassifier
snf.score(X_test,y_test) # 0.8571428571428571
snf.score(X_egitim,y_egitim) # 0.9473684210526315

- load_breast_cancer:
# KNeighborsClassifier
snf.score(X_egitim,y_egitim) 	# 0.946
snf.score(X_test,y_test)		# 0.937

- mglearn.datasets.make_wave
# KNeighborsRegressor
reg.score(X_egitim,y_egitim) # 0.8194343929538755
reg.score(X_test,y_test)	 # 0.8344172446249605

- load_boston
# LinearRegression
lr.score(X_egitim,y_egitim)	# 0.9520519609032732
lr.score(X_test,y_test)		# 0.6074721959665877
overfitting. çünkü aşırı açıklayıcı değişken var: ridge&lasso lazım:
# Ridge (alpha=0.1)
ridge01.score(X_egitim,y_egitim)# 0.9282273685001993
ridge01.score(X_test,y_test)	# 0.7722067936479815
# Lasso (alpha=0.01)
lasso001.score(X_egitim,y_egitim)	# 0.8962226511086496
lasso001.score(X_test,y_test)		# 0.7656571174549979

- student-mat.csv
# LinearRegression
linear.score(X_test,y_test)		# 0.8325898318712225
linear.score(X_egitim,y_egitim)	# 0.8261275475197141


### mglearn plotları:
