import matplotlib.pyplot as plt
__import__('matplotlib').interactive(True)
import numpy as np
################################## MODEL İŞLEMLERİ
rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)
plt.scatter(x,y)
# bu grafikten gördük ki x ile y arasında linear bir ilişki var
# ki zaten denklemden onu bekliyorduk
from sklearn.linear_model import LinearRegression
# belirleyeceğimiz parametrelere hiper parametre denir. onları yazmazsak default parametreler geçerli olur
model = LinearRegression(fit_intercept=True) # fit_intercept zaten True. neden yazdık anlamadım. belki tutorial'daki sürüm farklıdır
X = x[:,np.newaxis]
model.fit(X,y) 
model.coef_ #sklearn'de işler böyle gibi bişey söyledi
model.intercept_ # (modelin sabiti)
# model kurulduktan sonraki adım yeni verileri değerlendirmektir. bunun için predict metodunu kullanıcaz
x_fit = np.linspace(-1,11) # örnek bir girdi
X_fit = x_fit[:,np.newaxis] # predict metodu tek boyutlu arrayleri kabul etmiyo demekki
y_fit = model.predict(X_fit)
plt.figure()
plt.scatter(x,y)
plt.plot(x_fit,y_fit)
# ben niye bi cemaate bi fırkaya bi yere girmedim hürriyetim elimde olsun diye, istediğimi söyleyeyim diye
# yunan batı trakyada şapka, harf, mahkeme
# batı kanunları, hocaları asması, şeriatı lağvetmesi, hilafeti yıkması, tekkeyi kapatması
# 1000 yıllık cihan hakimiyeti olan bir millet ,sbviçrenin yirmide biri biri bir kantonun kanununu alıp da tatbik etmez
