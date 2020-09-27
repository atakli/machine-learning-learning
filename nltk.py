# coding: utf-8
import nltk
get_ipython().system('pip install nltk')
get_ipython().system('pip install nltk')
import nltk
gb = nltk.corpus.gutenberg
type(gb)
gb.fileids # hutenberg paketindeki dosyalar
# önce indirmem gerekiyodu:
nltk.downlaod('gutenberg')
nltk.download('gutenberg')
gb.fileids # hutenberg paketindeki dosyalar
gb.fileids() # hutenberg paketindeki dosyalar
hamlet = nltk.corpus.gutenberg.words('shakespeare-hamlet.txt')
len(hamlet) # bu metnin uzunluğu (bileşen sayısı)
# metindeki ilk on kelime:
hamlet[:10]
hamlet[:16]
# sent fonksiyonunh kullanırsak herbir cümle bir bileşen olarak alınır
hamlet_sents = nltk.corpus.gutenberg.sents('shakespeare-hamlet.txt')
hamlet_sents[:5]
type(hamlet)
# ilginç. sents'i çalıştırmak için bi download'a daha ihtiaç varmış, niyeyse
nltk.download('punkt')
hamlet_sents = nltk.corpus.gutenberg.sents('shakespeare-hamlet.txt')
hamlet_sents[:5]
# ilk beş cümleydi bu
# metin olaylarında en temel işlerden biri metinde kelime aramaktır
text = nltk.Text(hamlet)
text.concordance('Stage')
# stage kelimesinin önünde ve arkasında ne oluğunuı yazdıralım
text.common_contexts(['Stage'])
### KELİME SIKLIĞI ANALİZİ
fd = nltk.FreqDist(hamlet)
fd.most_common(10)
# virgül nokta bağlaç filan var genelde. bunlar genellikle çıkarılır
# nltk kütüphanesinde stub words vardır:
sw = set(nltk.corpus.stopwords.words('english'))
nltk.download('stopwords')
nltk.download('stopwords')
sw = set(nltk.corpus.stopwords.words('english'))
len(sw)
list(sw)[:10]
list(sw)[:10]
# tutorial'da farklı. değişmiş demekki sırası
hamlet.filtered = [w for w in hamlet if w.lower() not in sw]
r.e = 34
r = 3
r.e = 34
hamlet_filtered = [w for w in hamlet if w.lower() not in sw]
dir(hamlet)
get_ipython().run_line_magic('pinfo', 'hamlet.filtered')
jamlet.filtered
hamlet.filtered
get_ipython().run_line_magic('pinfo', 'hamlet.filtered')
# bakalım filtreden sonra  en çok kullanılan kelimeler neler
fd = nltk.FreqDist(hamlet_filtered)
fd.most_common(10)
# tmm stop words elenmiş ama noktalama işaretleri hâlâ duruyo
import string
p = set(string.punctuation)
hamlet_filtered = [w for w in hamlet if w.lower() not in sw and w.lower() not in p]
fd = nltk.FreqDist(hamlet_filtered)
fd.most_common(10)
# BIGRAMS VE TRIGRAMS
# tek kelime ile uğraşmak yerine bigrams denen kelime çiftlreiyle de çalışabiliriz
# fats food, take care gibi. bunlara collocation denir. bu arada tutor'umun "teyk ker" deyişi güzeldi :))
bgrms = nltk.FreqDist(nltk.bigrams(hamlet_filtered))
bgrms.most_common(15)
tgrms = nltk.FreqDist(nltk.trigrams(hamlet_filtered))
tgrms.most_common(15)
tgrms.most_common(10)
# DUYGU ANALİZİ
# yani sentiment. opinion mining de denir
# belli anahtar kelimelere fdayanarak kişilerin memnuniyet dereceleri buşlunur
# positif, negatif, nötr şeklinde üç duygu durumu vardır
from nltk.corpus import movie_reviews
documents = [list(movie_reviews.words(fileid)), category)
documents = [list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)
            
documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]
            
            
            
           
nltk.download('movie_reviews')
documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]
            
            
            
           
# bu çok uzun sürdü lan
# dökümanı karıştırmak için random modülünü kullanalım
import random
random.shuffle(documents)
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
type(all_words)
get_ipython().run_line_magic('pinfo', 'all_words.keys')
word_feature = list(all_words.keys())[:2000] # en çok kullanılan 2000 kelimeyi aldık (onların feature'leri heralde)
len(word_feature)
word_feature[0]
word_feature[:10]
all_words[:5]
(all_words)[:5]
list(all_words)[:5]
# keys ne anlamadık
# m
def document_features(document): 3 belirlediğimiz kelimelerin dökümanda olup olmadığını kontrol ediyor
def document_features(document): # belirlediğimiz kelimelerin dökümanda olup olmadığını kontrol ediyor
    document_words = set(document)
    features = {}
    for word in word_feature:
        features['contains(%s)' % word] = (word in document_words)
    return features
    
featuresets = [(document_features(d),c) for (d,c) in documents]
len(featuresets)
