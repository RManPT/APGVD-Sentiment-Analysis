
# coding: utf-8

# <a id="topo"></a>
# ___
# <img src="../_images/logo_mei.jpg" alt="Mestrado em Internet das Coisas @ IPT" width="200"/>
# <div class="alert alert-block alert-success" align="center">
# <h1>Análise e Processamento de Grandes Volumes de Dados</h1>
# <h3>Sentiment analysis - 3ª parte</div>
# <center><h5>Criado por: Bruno Bernardo / David Carrilho / Rui Rodrigues</h5></center>
# ___
# 
# [<img src="../_images/download.jpg" alt="Mestrado em Internet das Coisas @ IPT" width="50"/>](Sentiment.ipynb)
# ___
# 
# **Crédito para Peter Nagy February 2017** 
# https://github.com/nagypeterjob

# [Análise sentimental supervisionada (Naive Bayes)](#super)<br>
# [Preparação](#preparacao)<br>
# [Importação](#import)<br>
# [Dataset](#dataset)<br>
# [Importação dos dados](#import2)<br>
# [Separação de dados em treino e teste](#split)<br>
# [Visualização dos dados de treino](#visual)<br>
# [Preparação subset de treino](#treino)<br>
# [Extração de características](#features)<br>
# [Classificação](#classificacao)<br>
# [Conclusão](#conclusao)

# <p>&nbsp;</p>
# <p>&nbsp;</p>
# <p>&nbsp;</p>
# 
# <a id="super"></a>
# # Análise sentimental supervisionada (Naive Bayes)
# <p>&nbsp;</p>
# 
# Depois de termos demonstrado como obter análise de sentimento sobre ficheiros e <i>feeds</i> usando uma API vamos agora demonstrar como usar directamente o [NLTK](#http://www.nltk.org/) (<i>Natural Language Tool Kit</i>) para treinar classificadores estatísticos e posteriormente usá-los para fazer a análise sentimental. Para isto usamos datasets recolhidos de [CrowdFlower](#https://www.crowdflower.com/data-for-everyone/). Foram escolhidos dois datasets com apenas 3 classes mas este método permite usa mais classes.
# 
# Neste primeiro exemplo vamos usar o classificador NB. Este classificador assume que as características dos vectores são independentes. Quando esta assumpção se verifica este é um classificador de alta precisão. É o que vamos avaliar.
# 

# ------

# <a id="preparacao"></a>
# ## Preparação
# 
# Se correr o notebook no seu próprio server Jupyter deverá instalar dependências.
# 
# Na "Anaconda Comand Prompt" (na pasta onde tens os notebooks):
# 
#     pip install numpy
# 
#     pip install pandas
# 
#     pip install nltk
# 
#     pip install wordcloud
#     
#     
# <div class="alert alert-block alert-info">NOTA: Se não tiveres premissões para instalar, executa o Prompt como Administrador. Se estiveres da drive C: e os notebooks estiverem na D:, assim que abres o Prompt deves escrever D: para mudar de drive.</div>
# 
# <div class="alert alert-block alert-danger">Dependendo da vossa versão do Python poderão ter dificuldade em instalar as dependências do Wordcloud. Se for esse o caso façam o download da versão adequada [daqui](#https://www.lfd.uci.edu/%7Egohlke/pythonlibs/#wordcloud) e executem:
#     pip install XXXXX sendo XXXXX o nome do ficheiro descarregado.</div>    
# 
# No nosso caso vamos usar um contentor dinâmico que permite a partilha de notebooks com edição, [mybinder.org](https://mybinder.org).
# 
# 

# <a id="import"></a>
# ### A importação dos módulos necessários

# In[ ]:


# algebra linear
import numpy as np 
# processamento de dados e I/O ficheiros 
import pandas as pd 
# função que separa os dados em uma percentagem para treino e outra para teste
from sklearn.model_selection import train_test_split 
# Natural language tool kit
import nltk
# biblioteca de stopwords
from nltk.corpus import stopwords
# biblioteca de machine learning
from nltk.classify import SklearnClassifier
# gerador de nuvens de palavras
from wordcloud import WordCloud,STOPWORDS
# manipulação de imagens
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
# biblioteca de verificação de ficheiros
from subprocess import check_output


# <a id="dataset"></a>
# ### O Dataset
# 
# Preparámos dois datasets:
# 
#     * Judge emotions about nuclear energy from Twitter (2013)
# 
#     É uma colecção de tweets relacionados com energia nuclear juntos com a classificação feita por utilizadores do sentimento relacionado com o tweet. As categorias disponíveis são: "Positivo", "Neutro" e "Negativo". O dataset contém ainda informação sobre o nível de confianção numa categorização correcta.
#     
#     * First GOP debate sentiment analysis (2015)
# 
#     É uma colecção de dezenas de milhares de tweets sobre o debate dos candidatos presidenciais republicanos dos EUA. Aos utilizadores foi pedido que categorizassem a nível de sentimento, a relevância, candidato a que se refere e nível de confiança na categorização correcta.
# 
# Foram escolhidos estes dois precisamente por usarem apenas 3 classes, o que se adequa à nossa demonstração. 

# <a id="import2"></a>
# ### Importação dos dados
# 
# Os nossos <i>dataset</i> são CSVs que contém diversa informação, como o nosso objectivo é fazer análise sentimental precisamos apenas das colunas com o texto do tweet e a classificação atribuida por utilizadores. 

# In[ ]:


data = pd.read_csv('../_datasets/GOP_REL_ONLY.csv', encoding = "ISO-8859-1")
#data = pd.read_csv('../_datasets/1377191648_sentiment_nuclear_power.csv', encoding = "ISO-8859-1")
# Keeping only the neccessary columns
data = data[['text','sentiment']]


# <a id="split"></a>
# ### Separação de dados em treino e teste
# 
# Primeiro que tudo vamos dividir uma percentagem (10%) do dataset para treino e usar o restante para  testes. Para a fase de treino optamos por abdicar da categoria "Neutral" pois o que pretendemos é fazer a diferenciação entre positivos e negativos.
# 

# In[ ]:


# Divisão dos dados entre treino e teste
train, test = train_test_split(data,test_size = 0.1)
# Exclusão do treino dos tweets de categoria neutra
train = train[train.sentiment != "Neutral"]


# <a id="visual"></a>
# ### Visualização dos dados de treino
# 
# Achámos que seria interessante visualizar quais as palavras mais marcantes de cada categoria, então separámos no subset de treino os tweets positivos dos negativos.

# In[ ]:


train_pos = train[ train['sentiment'] == 'Positive']
train_pos = train_pos['text']
train_neg = train[ train['sentiment'] == 'Negative']
train_neg = train_neg['text']


# #### Preparação da cloud
# 
# Como só pretendemos usar as palavras vamos fazer uma cópia de cada categoria e limpar os <i>tweets</i> de links, nomes de utilizadores, <i>hashtags</i> e indicações de <i>retweet</i>. Posteriormente temos de remover as <i>stopwords</i> (definidas pela <i>WordCloud</i>) e preparar a <i>WordCloud</i>. Juntámos todas estas instruções num único método para poder ser reutilizado para ambas as categorias: 

# In[ ]:


def wordcloud_draw(data, color = 'black'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and not word.startswith('#')
                                and word != 'RT'
                            ])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()


# #### WordCloud de palavras positivas

# In[ ]:


print("Palavras positivas")
wordcloud_draw(train_pos,'white')


# #### WordCloud de palavras negativas

# In[ ]:


print("Palavras negativas")
wordcloud_draw(train_neg)


# #### Reflexão sobre o significado
# 
# Será interessante procurar entender pela <i>WordCloud</i>, particularmente pelas palavras mais frequentes (as de tamanho de fonte maior) entender a inclinação dos utilizadores sobre o assunto. Como terão sido usadas estas palavras nos seus <i>tweets</i>? Que entendimento poderemos retirar desta disposição da informação?

# <a id="treino"></a>
# ### Preparação subset de treino
# 
# Agora vamos fazer efectivar no <i>subset</i> de treino a limpeza que tinhas efectuado para a <i>WordCloud</i> mas desta vez usando as <i>stopwords</i> do NLTK.
# 
# 
# <div class="alert alert-block alert-success">**Stop Word:** Stop Words são palavras sem impacto significativo nas pesquisas. Por serem muito comuns retornam uma enorme quantidade de resultados desnecessários e por isso são removidas. (ex: the, for, this etc.)</div>

# #### Descarregar as stopwords adequadas
# 
# Sendo que o nosso <i>dataset</i> está em inglês devemos efectuar o <i>download</i> das <i>stop words</i> adequadas:

# In[ ]:


tweets = []
# download e descompactação do "corpus" de stop words
nltk.download("stopwords")
# selecção da língua
stopwords_set = set(stopwords.words("english"))


# #### Iterar no set de treino e fazer a limpeza

# In[ ]:


for index, row in train.iterrows():
    words_filtered = [e.lower() for e in row.text.split() if len(e) >= 3]
    words_cleaned = [word for word in words_filtered
        if 'http' not in word
        and not word.startswith('@')
        and not word.startswith('#')
        and word != 'RT']
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
    tweets.append((words_cleaned,row.sentiment))


# #### Obtenção dos vectores limpos

# In[ ]:


test_pos = test[ test['sentiment'] == 'Positive']
test_pos = test_pos['text']
test_neg = test[ test['sentiment'] == 'Negative']
test_neg = test_neg['text']


# <a id="features"></a>
# ## Extração de características (features)
# 
# Seguidamente criamos o chamado modelo <i>[Bag of Words](#https://en.wikipedia.org/wiki/Bag-of-words_model)</i> através da tokenização e a contagem da frequência das palavras. Basicamente este modelo é uma representação numérica do nosso texto e baseado nessa representação podem ser retiradas muitas informaçãos. É deste <i>Bag of words</i> que vão ser gerados os vectores que alimenentam a BM (<i>Bayes machine</i>). Para isto vamos usar mais uma vez o NLTK para calcular as frequências e definir as chaves.

# #### Função para obter as palavras em cada tweet

# In[ ]:


# Função para obter as palavras em cada tweet
def get_words_in_tweets(tweets):
    all = []
    for (words, sentiment) in tweets:
        all.extend(words)
    return all


# #### Função para obter as features de cada conjunto colecção de palavras

# In[ ]:


# Função para obter as features de cada conjunto colecção de palavras
def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    features = wordlist.keys()
    return features


# #### Por fim, obtenção das features

# In[ ]:


# Obtenção das features da colecção de palavras encontradas nos tweets
w_features = get_word_features(get_words_in_tweets(tweets))

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in w_features:
        features['containts(%s)' % word] = (word in document_words)
    return features


# #### WordCloud das features
# 
# Por curiosidade vamos fazer o <i>plot</i> de uma <i>wordcloud</i> das features do nosso <i>dataset</i>, ou seja as palavras mais frequentemente distrubuidas.

# In[ ]:


print("Features")
wordcloud_draw(w_features)


# <a id="classificacao"></a>
# ## Classificação
# 
# Usando o classificador NaiveBayes vamos finalmente classificar as <i>features</i>.

# #### Efectivação do treino

# In[ ]:


# Treino do classificador NaiveBayes
training_set = nltk.classify.apply_features(extract_features,tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)


# Finally, with not-so-intelligent metrics, I tried to measure how the classifier algorithm scored.

# In[ ]:


neg_cnt = 0
pos_cnt = 0
for obj in test_neg: 
    res =  classifier.classify(extract_features(obj.split()))
    if(res == 'Negative'): 
        neg_cnt = neg_cnt + 1
for obj in test_pos: 
    res =  classifier.classify(extract_features(obj.split()))
    if(res == 'Positive'): 
        pos_cnt = pos_cnt + 1
        
print('[Negative]: %s/%s '  % (len(test_neg),neg_cnt))        
print('[Positive]: %s/%s '  % (len(test_pos),pos_cnt))    


# <a id="conclusao"></a>
# ## Conclusão
# 
# O objectivo desde notebook era demonstrar como ir mais "directo à fonte", escolhendo um método de classificação, fazendo o seu treino e posterior teste.
# 
# Aparentemente, com os datasets disponíveis, funciona bem para tweets de conotação negativa. As dificuldades acontecem quando à ambíguidade, frases irónicas ou sarcásticas, o que seria de esperar pois logo no início referimos que o NB é mais eficaz quando à clara distinção de classes.
# 
# Vamos procurar concluir se outros métodos de classificação poderão obter melhores resultados. No próximo exercício vamos usar uma rede neural recorrente, com memória, a LSTM, para compararmos os resultados.
# 
# Outra nota de destaque é que o classificador NB é de extrema lentidão.

# [<div align="right" class="alert alert-block"><img src="../_images/top.png" width="20"></div>](#topo)
