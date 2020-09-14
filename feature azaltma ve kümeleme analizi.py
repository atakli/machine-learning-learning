# denetimsiz (unsupervised) öğrenme problemi olarak iris veri setinin boyutunu (öznitelik, feature) azaltma
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

### Burdan sonrası başka bir yöntem: kümeleme analizi
# bu da denetimsiz öğrenme algoritması
# bu herhangi bir etiket referansı olmayan grupları ayırmak için kullanılır
# güçlü bir kümeleme modeli olan GaussianMixture'ı kullanıcaz
model = GaussianMixture(n_components=3, covariance_type='full')
model.fit(X_iris) # dikkat ettiysen, denetimsiz öğrenme modeli olduğu için y_hedef değişkenini burada kıullanmadık
y_gmm = model.predict(X_iris)
iris['kumeleme'] = y_gmm
iris.head()
sns.lmplot('PCA1','PCA2',hue='species',data=iris, col='kumeleme',fit_reg=True)