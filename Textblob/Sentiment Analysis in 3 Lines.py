
# coding: utf-8

# <a id="topo"></a>
# ___
# <img src="../_images/logo_mei.jpg" alt="Mestrado em Internet das Coisas @ IPT" width="200"/>
# <div class="alert alert-block alert-success" align="center">
# <h1>Análise e Processamento de Grandes Volumes de Dados</h1>
# <h3>Sentiment analysis - 1ª parte</div>
# <center><h5>Criado por: Bruno Bernardo / David Carrilho / Rui Rodrigues</h5></center>
# ___
# 
# [<img src="../_images/download.jpg" alt="Mestrado em Internet das Coisas @ IPT" width="50"/>](Sentiment Analysis in 3 Lines.ipynb)
# ___

# [Análise Sentimental em 3 Linhas](#3linhas)<br>
# [Um passo à frente](#umpasso)<br>
# [Brincadeiras](#brinca)<br>

# <p>&nbsp;</p>
# <p>&nbsp;</p>
# <p>&nbsp;</p>
# 
# <a id="3linhas"></a>
# # Análise Sentimental em 3 Linhas
# <p>&nbsp;</p>
# 
# Para exemplificar a extracção de informação sentimental de um texto vamos usar uma API de processamento de linguagem natural, [Textblob](http://textblob.readthedocs.io/en/dev/quickstart.html#create-a-textblob).
# 
# Esta API permite-nos, entre outras:
#     <ul><li><b>Noun phrase extraction</b></li>
#             Identificação de entidades e seus modificadores (adjectivos, pronomes, etc). Permite fazer contextualização<br/>
#     <li><b>Part-of-speech tagging</b></li>
#     Permite relacionar definição e contexto de cada palavra com as que a rodeiam<br>
#     <li><b>Tokenization</b></li>
#     Partição de texto em frases e palavras<br>
#     <li><b>Word and phrase frequencies</b></li>
#     Criação de dicionários<br>
#     <li><b>Word inflection (pluralization and singularization) and lemmatization</b></li>
#     Permite reduzir o vocabulário de um texto à sua raiz, reduzindo em muitoo tamanho e complexidade de dicionários<br>
#     <li><b>Classification (Naive Bayes, Decision Tree)</b></li>
#     Classificaçõa probabilística ou baseada em grafos em que cada nó representa uma decisão e cada folha a probablilidade de um resultado<br>
#     <li><b>Sentiment analysis</b></li>
#     Dando uso de todas as anteriores podemos ainda obter informação sobre opiniões<br>
# </ul>

# <p></p>

# ## Sentiment analysis - resultados
# 
#    Tipicamente os resultados são exprimidos com dois indicadores:
# 
#    <b>Polaridade :</b> [-1 a 1]
#    Representa a emoção expressada pela frase. De forma simplista, pode ser positiva, negativa ou neutra.
#     
#    -1 totalmente negativo / 0 neutro / 1 totalmente positivo
# 
#    <b>Subjetividade:</b> 0 a 1
#    O texto por si representa ou não uma opinião/especulação. Tem de ser contextualizado pois pode influenciar a informação. 
#     
#    0 imparcialidade-objectividade / 1 o autor influencia a extração de informação
#    

# 
# ------
# 

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

# ### A importação dos módulos necessários

# In[ ]:


from textblob import TextBlob


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# ### As 3 linhas

# Análise de uma string:

# In[ ]:


wiki = TextBlob("I love you very much. I am happy.")


# Obtenção dos resultados:

# In[ ]:


polaridade = wiki.sentiment.polarity
subjetividade  = wiki.sentiment.subjectivity


# Formatação e plot:

# In[ ]:


print("Polaridade: "+"{:.2}".format(polaridade));
print("Subjetividade : "+"{:.2}".format(subjetividade ));


# Interpretação visual da polaridade:

# In[ ]:


if polaridade > 0.33:
    img=mpimg.imread('../_images/pos.jpg')
    imgplot = plt.imshow(img)
    plt.show()
if -0.33 <= polaridade <= 0.33:
    img=mpimg.imread('../_images/neu.jpg')
    imgplot = plt.imshow(img)
    plt.show()
if polaridade < -0.33:
    img=mpimg.imread('../_images/neg.jpg')
    imgplot = plt.imshow(img)
    plt.show()


# <p>&nbsp;</p>
# <a id="umpasso"></a>
# # Um passo à frente 

# ### Usar um ficheiro de texto como fonte de dados...
# 

# <small>Ficheiros:
# 
# data1.txt : Primeiro Livro da Biblia<br>
# data2.txt : Declaração da Independecia dos Estados Unidos da América<br>
# data3.txt : Discurso de Trump na ONU 2017<br>
# data4.txt : Carta de amor<br>
# data5.txt : Discurso Oprah</small>
# 

# <p>&nbsp;</p>

# Conectar ao ficheiro:

# In[ ]:


with open('data5.txt','r', encoding = "ISO-8859-1") as myfile:
    data=myfile.read().replace('\n', '')


# Extrair dados:

# In[ ]:


wiki = TextBlob(data)


# Obtenção de resultados:

# In[ ]:


polaridade = wiki.sentiment.polarity
subjetividade  = wiki.sentiment.subjectivity


# Formatação e plot:

# In[ ]:


print("Polaridade: "+"{:.2}".format(polaridade));
print("Subjetividade : "+"{:.2}".format(subjetividade ));


# <p>&nbsp;</p>

# <a id="brinca"></a>
# # Brincadeiras

# Como influenciar resultados:

# In[ ]:


wiki = TextBlob("I hate")
#wiki = TextBlob("I hate when bad thing happen")
#####################################################
#wiki = TextBlob("I hate dogs")
#wiki = TextBlob("I hate bad dogs")
#wiki = TextBlob("I hate bad angry dogs")
#####################################################
#wiki = TextBlob("I blipKING HATE YOU LITTLE blIT")


# Mais do mesmo:

# In[ ]:


polaridade = wiki.sentiment.polarity
subjetividade  = wiki.sentiment.subjectivity

print("Polaridade: "+"{:.2}".format(polaridade));
print("Subjetividade : "+"{:.2}".format(subjetividade ));


# Interpretação visual mais detalhada:

# In[3]:


if polaridade > 0.6:
    img=mpimg.imread('../_images/awesome.jpg')
    imgplot = plt.imshow(img)
    plt.show()
if 0.20 <= polaridade <= 0.6:
    img=mpimg.imread('../_images/good.jpg')
    imgplot = plt.imshow(img)
    plt.show()
if -0.20 < polaridade < 0.2:
    img=mpimg.imread('../_images/neutral.jpg')
    imgplot = plt.imshow(img)
    plt.show()
if -0.60 <= polaridade <= -0.2:
    img=mpimg.imread('../_images/bad.jpg')
    imgplot = plt.imshow(img)
    plt.show()
if polaridade < -0.6:
    img=mpimg.imread('../_images/horrible.jpg')
    imgplot = plt.imshow(img)
    plt.show()


# [<div align="right" class="alert alert-block"><img src="../_images/top.png" width="20"></div>](#topo)
