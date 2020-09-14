import mglearn
mglearn.plots.plot_linear_regression_wave()
__import__('matplotlib').interactive(True)
from sklearn.linear_model import LinearRegression
X,y = mglearn.datasets.make_wave(n_samples=60)
from sklearn.model_selection import train_test_split
X_egitim, X_test, y_egitim, y_test = train_test_split(X,y,random_state=42)
lr = LinearRegression().fit(X_egitim,y_egitim)
lr.coef_ # katsayı, yani eğim # öznitelik sayılarına göre numpy dizisidir
lr.intercept_ # sabit # herzaman ondalıktır
# sklearn'de eğitim verisinden üretilen parametlerin sonuna alt tire eklenir. böylece bulunan parametreler ile kullanıcı tarafından yazılan parametreler birbirinden ayrılır
lr.score(X_egitim,y_egitim)
lr.score(X_test,y_test)
# çok yakın çıktığına göre yüksek olasılıkla burda overfitting yok ama model basit olduğundan underfitting var
# bu, sonuç değişkenini açıklamak için başka değişkenlere ihtiyaç var anlamına gelir
### ÇOKLU LİNEER REGRESYON
from sklearn.datasets import load_boston
boston = load_boston()
print(boston['DESCR'])
X,y = mglearn.datasets.load_extended_boston()
X.shape
# 104, 13 tane öznitelik sayısını ve bu özniteliklerin ikişerli ilişkilerini gösteriyor. nee?
y.shape
X_egitim, X_test, y_egitim, y_test = train_test_split(X,y,random_state=0)
lr = LinearRegression().fit(X_egitim,y_egitim)
lr.score(X_egitim,y_egitim)
lr.score(X_test,y_test)
# bu tutarsızlık overfitting, yani aşırı uydurmaya işaret ediyor
# aşırı açıklayıcı değişken var, yani model kompleksliği fazla
# model kompleksliğini kontrol etmek için standart linear regresyonun alternatifi olan 
# rich ve rasso regresyonlar kullanılır. bunlar sonra gelecek
### UYGULAMA
import pandas as pd
veri = pd.read_csv('student/student-mat.csv',sep=';')
veri = veri[['G1','G2','G3','studytime','failures','absences','age']]
# veri setindeki istediğimiz öznitelikleri seçmiş olduk
veri.head()
# G dediği grade, studytime haftalık çalışma zamanı, failure sınıf tekrarı
# öznitelikler rename edilebilir
veri.rename(columns={'G1':'Not1',
                     'G2':'Not2',
                     'G3':'Final',
                     'studytime':'Calisma_sure',
                     'failures':'Sinif_tekrari',
                     'absences':'Devamsizlik',
                     'age':'Yas'},inplace=True) # inplace argümanı yaptığımız değişikliği veri setinde korumayı sağlar
veri.head()
veri.dtypes
#sklearn'de model oluştururken veriyi np dizisine çevirmek gerekirimport numpy as np
y = np.array(veri['Final'])
# şimdi girdi verisini oluşturalım:
X = np.array(veri.drop('Final',axis=1))
from sklearn.model_selection import train_test_split
# test_size argümanı veri setini hangi oranda böleceğimizi belirler
X_egitim, X_test, y_egitim, y_test = train_test_split(X,y,test_size=0.2,random_state=2)
from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(X_egitim,y_egitim)
linear.score(X_test,y_test)
linear.score(X_egitim,y_egitim)
# test ile eğitim verisindeki doğruluk oranlarının birbirine yakın olması beklenir
# test verisindeki doğruluk oranı yüksek diğeri düşükse underfitting, tersi ise overfitting var demektir
# burda az biraz düşük uydurma var diyebiliriz
print('Katsayılar: \n',linear.coef_)
print('Sabit:\n',linear.intercept_)
veri.head()
# Katsayılar sırasıyla sütunların katsayıları. mesela 0.19575962 Not1'in katsayısı
yeni_veri = np.array([[10,14,3,0,4,16]]) # uydurduk bu veriyi, buna göre final notunu bulmak için
linear.predict(yeni_veri)
# şimdi deneme amaçlı, sonucunu bildğimiz, veri setinde yer alan bir öğrencinin verilerini girelim:
yeni_veri2 = np.array([[15,14,3,0,2,15]])
linear.predict(yeni_veri2)
# gerçek değer olan 15'e yakın çıktı, fena değil