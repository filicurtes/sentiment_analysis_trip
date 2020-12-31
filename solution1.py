from sklearn.svm import LinearSVC
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from nltk.corpus import stopwords as sw
import spacy
from spacy.tokens import Doc
from nltk.tokenize import word_tokenize
import string  

df=pd.read_csv('development.csv')
df_ev=pd.read_csv('evaluation.csv')
y=df['class'].tolist()
reviews=df.text.tolist()
reviews_ev=df_ev.text.tolist()

male_name=[' marco', 'alessandro', 'giuseppe', 'flavio', 'luca', 'giovanni', 'roberto', 'andrea', 'stefano', 'francesco', 'mario', 'luigi']
female_name=['anna', 'maria', 'sara', 'laura', 'aurora', 'valentina', 'giulia', 'rosa', 'gianna', 'giuseppina', 'giovanna', 'sofia']

class MyTokenizer(object):
    def __init__(self):
        self.nlp=spacy.load('it_core_news_sm',disable=['parser','ner'])
        self.translator=str.maketrans(string.punctuation,' '*len(string.punctuation),'')
    def __call__(self,document):
        to_lem=[]
        lemmatized=[]
        document=document.translate(self.translator)
        for w in word_tokenize(document):
            w=w.strip()
            w=w.lower()
            if  not w.isdigit():
                to_lem.append(w)
        doc=Doc(self.nlp.vocab,to_lem)
        for t in doc:
            lemma=t.lemma_
            ##if lemma not in self.stop_words:
            lemmatized.append(lemma)
        return lemmatized

stop_words=sw.words('italian')
stop_words=stop_words+male_name
stop_words=stop_words+female_name
stop_words=stop_words+['marcare','rodere','camera','venezia','hotel','roma','firenze','bologna']
stopwords_to_keep=['non','contro','perché','più']
for x in stopwords_to_keep:
    stop_words.remove(x)

nlp=spacy.load('it_core_news_sm',disable=['parser','ner','tagger'])
document=Doc(nlp.vocab,stop_words)
stopwords_lemm=[]
for w in document:
        lemma=w.lemma_
        stopwords_lemm.append(lemma)
stopwords_lemm=list(dict.fromkeys(stopwords_lemm))
mytokenizer=MyTokenizer()

tfidfvectorizer=TfidfVectorizer(tokenizer=mytokenizer,ngram_range=(1,3),stop_words=stopwords_lemm,min_df=0.0001)
X=tfidfvectorizer.fit_transform(reviews)
X_test=tfidfvectorizer.transform(reviews_ev)

lin_svc=LinearSVC(max_iter=30000)
lin_svc.fit(X,y)
y_pred=lin_svc.predict(X_test)

df_submit=pd.DataFrame(columns=['Id','Predicted'])
for i,el in enumerate(y_pred):
    df_submit.loc[i,'Id']=i
    df_submit.loc[i,'Predicted']=y_pred[i]
df_submit.to_csv('Results.csv',index=False)