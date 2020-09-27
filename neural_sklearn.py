# neural network ufak bi probelm yerine onn yerine knn yada linearregression'un polynamial regresyon gibi daha farklı modellerini 
# kullanmak daha mantıklı

# yüklediğimiz modüller tamamen knn'le aynı; tek farkı mlp regressor

# normalde In7'de parantez içini yazmaya gerek yok. 35yerine 200 olsaydı gerek kalmazdı?
# 35 yerine 200 tane olsaydı çok daha başarılı bişekilde verimizi çözerdi

# verbose 0 çünkü her aşama için çıktı almak istremedim
# random_state=0 sistemi oluştutrurken aoluşturduğum seçkisiz ağırlıkları sabit tuttum. her kayıtta farklı bişey bulunmayacak. ya da
# siz bunu denediğiniz zmaan bu veri seiyle herşiey dopru yaptıysanız aynı somnucu bulucasınız

# biz network'umuzu ilk oluiturduğumuzda bunlar seçkisiz olarka sitem olarak oluşturlulyo
# 3 dökümantasyonu okursan bunlara da müdahale edebilirsin

# ilk katmandan katsayılarla çarpılarak gelen sayıları nöron bir fonksiyondan geçirdi:  aktivasyon fonksiyonu: çeşitleri var:
# sigmoid, relü, elü. sonucun oluştuğu yere de bir aktivasyon fonksiyonu koyulabiliyo

# epoc = iterasyon