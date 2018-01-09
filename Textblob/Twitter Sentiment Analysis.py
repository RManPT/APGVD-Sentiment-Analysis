
# coding: utf-8

# <a id="topo"></a>
# ___
# <img src="../_images/logo_mei.jpg" alt="Mestrado em Internet das Coisas @ IPT" width="200"/>
# <div class="alert alert-block alert-success" align="center">
# <h1>Análise e Processamento de Grandes Volumes de Dados</h1>
# <h3>Sentiment analysis - 2ª parte</div>
# <center><h5>Criado por: Bruno Bernardo / David Carrilho / Rui Rodrigues</h5></center>
# ___
# 
# [<img src="../_images/download.jpg" alt="Mestrado em Internet das Coisas @ IPT" width="50"/>](Twitter Sentiment Analysis.ipynb)
# ___

# [Análise sentimental no Twitter](#twitter)<br>
# [Preparação](#preparacao)<br>
# [Importação](#import)<br>
# [Autenticação](#auth)<br>
# [Parâmetros](#def)<br>
# [Pesquisa](#search)<br>
# [Resultados individuais](#resi)<br>
# [Sentimento generalizado](#rest)

# <p>&nbsp;</p>
# <p>&nbsp;</p>
# <p>&nbsp;</p>
# 
# <a id="twitter"></a>
# # Análise sentimental no Twitter
# <p>&nbsp;</p>
# 
# As redes sociais são a maior fonte de informação e categorização. Os utentes/matéria prima destas redes tanto expressam as suas opiniões como categorizam, pela emoção que sentem, as opiniões de outros sobre uma infinidade de assuntos.
# 
# Faz todo o sentido explorar esta fonte infindável de informação.
# 
# Neste exercício vamos demonstrar como aceder a feeds do Twitter e deles extrair informação sentimental.
# 

# ------

# <a id="preparacao"></a>
# ## Preparação
# 
# Se correr o notebook no seu próprio server Jupyter deverá instalar dependências.
# 
# Na "Anaconda Comand Prompt" (na pasta onde tens os notebooks):
# 
#     pip install tweepy
# 
#     pip install textblob
# 
# <div class="alert alert-block alert-info">NOTA: Se não tiveres premissões para instalar, executa o Prompt como Administrador. Se estiveres da drive C: e os notebooks estiverem na D:, assim que abres o Prompt deves escrever D: para mudar de drive.</div>
# 
# No nosso caso vamos usar um contentor dinâmico que permite a partilha de notebooks com edição, [mybinder.org](https://mybinder.org).

# <p>&nbsp;</p>

# <a id="import"></a>
# ### A importação dos módulos necessários

# In[159]:


import tweepy
from textblob import TextBlob


# In[160]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# <a id="auth"></a>
# ### Autenticação à API do Twitter

# In[161]:


consumer_key= 'nEiGfOnHr3hcIgQ0k08kG19q2'
consumer_secret= 'tnbh1kfHYFff3PS0c6AjXlPqYvL2lJG8wdej5HZvpqvDH6Ybfr'

access_token='776640421-r3L3A1YaiMGlJrqlPE08wCInKIIvzN2WoxsyLMZx'
access_token_secret='vDNBDiLyLeu0IcjylH7liqpzW8OM5sohNSdl9MsVnjeok'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)


# <a id="def"></a>
# ### Definição de parâmetros de pesquisa

# In[162]:


query = 'crypto'
max_tweets = 100
lang = 'en'


# <a id="search"></a>
# ### Efectuar pesquisa
# 
# Devido a limites nas pesquisas vamos usar um mecanismo para obter os twetts em parcelas até termos o total definido:

# In[163]:


public_tweets = []
last_id = -1
while len(public_tweets) < max_tweets:
    count = max_tweets - len(public_tweets)
    try:
        new_tweets = api.search(q=query, count=count, max_id=str(last_id - 1), lang='en')
        if not new_tweets:
            break
        public_tweets.extend(new_tweets)
        last_id = new_tweets[-1].id
    except tweepy.TweepError as e:
        # depending on TweepError.code, one may want to retry or wait
        # to keep things simple, we will give up on an error
        break


# <a id="resi"></a>
# ### Resultados Individuais
# 
# Agora vamos excluir dos tweets obtidos todos aqueles que são retweets e hyperlinks do conteúdo dos que sobram. Seguidamente imprimimos os tweets, a análise sentimental e respectivo emoticon.

# In[164]:


count=0
totalPol=0
totalSub=0

for tweet in public_tweets:
    # Excluir retweets
    if (not tweet.retweeted) and ('RT @' not in tweet.text):
        # Remover Sites dos Tweets
        text = re.sub(r'http\S+', '', tweet.text, flags=re.MULTILINE)
        print(str(count+1)+'/'+str(len(public_tweets)))
        print(text)

        #Executar Analise Sentimental no Tweet
        analysis = TextBlob(text)

        #Auxiliares para Resultados Totais
        count +=1
        totalPol += analysis.sentiment.polarity
        totalSub += analysis.sentiment.subjectivity

        print(analysis.sentiment)
        print("")
        plt.figure(figsize=(1,1))
        # plot emoticon
        if analysis.sentiment.polarity > 0.6:
            img=mpimg.imread('../_images/awesome.jpg')
            imgplot = plt.imshow(img)
            plt.show()
        if 0.20 <= analysis.sentiment.polarity <= 0.6:
            img=mpimg.imread('../_images/good.jpg')
            imgplot = plt.imshow(img)
            plt.show()
        if -0.20 < analysis.sentiment.polarity < 0.2:
            img=mpimg.imread('../_images/neutral.jpg')
            imgplot = plt.imshow(img)
            plt.show()
        if -0.60 <= analysis.sentiment.polarity <= -0.2:
            img=mpimg.imread('../_images/bad.jpg')
            imgplot = plt.imshow(img)
            plt.show()
        if analysis.sentiment.polarity < -0.6:
            img=mpimg.imread('../_images/horrible.jpg')
            imgplot = plt.imshow(img)
            plt.show()

        


# <a id="rest"></a>
# ### Resultados Totais
# 
# Fazendo uma simples média aritmética tentamos estimar de forma rude o sentimento geral sobre o alvo da nossa pesquisa:

# In[165]:


polaridadeTotal = (totalPol/count)
subjetividadeTotal = (totalSub/count)

print("Polaridade: "+"{:.2}".format(polaridadeTotal));
print("Subjetividade : "+"{:.2}".format(subjetividadeTotal ));

if polaridadeTotal > 0.6:
    img=mpimg.imread('../_images/awesome.jpg')
    imgplot = plt.imshow(img)
    plt.show()
if 0.20 <= polaridadeTotal <= 0.6:
    img=mpimg.imread('../_images/good.jpg')
    imgplot = plt.imshow(img)
    plt.show()
if -0.20 < polaridadeTotal < 0.2:
    img=mpimg.imread('../_images/neutral.jpg')
    imgplot = plt.imshow(img)
    plt.show()
if -0.60 <= polaridadeTotal <= -0.2:
    img=mpimg.imread('../_images/bad.jpg')
    imgplot = plt.imshow(img)
    plt.show()
if polaridadeTotal < -0.6:
    img=mpimg.imread('../_images/horrible.jpg')
    imgplot = plt.imshow(img)
    plt.show()


# [<div align="right" class="alert alert-block"><img src="../_images/top.png" width="20"></div>](#topo)

# ### Ref.
# 
# https://youtu.be/o_OZdbCzHUA
# 
# https://github.com/llSourcell/twitter_sentiment_challenge
# 
# https://github.com/kalradivyanshu/TwitterSentiment
# 
# http://docs.tweepy.org/en/v3.5.0/api.html
# 
