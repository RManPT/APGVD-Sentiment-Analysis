
# coding: utf-8

# <a id="topo"></a>
# ___
# <img src="../_images/logo_mei.jpg" alt="Mestrado em Internet das Coisas @ IPT" width="200"/>
# <div class="alert alert-block alert-success" align="center">
# <h1>Análise e Processamento de Grandes Volumes de Dados</h1>
# <h3>Sentiment analysis - 4ª parte</div>
# <center><h5>Criado por: Bruno Bernardo / David Carrilho / Rui Rodrigues</h5></center>
# ___
# 
# [<img src="../_images/download.jpg" alt="Mestrado em Internet das Coisas @ IPT" width="50"/>](lstm.ipynb)
# ___
# 
# **Crédito para Peter Nagy February 2017** 
# https://github.com/nagypeterjob

# [Análise sentimental supervisionada (Naive Bayes)](#super)<br>
# [Preparação](#preparacao)<br>
# [Importação de módulos](#import)<br>
# [Dataset](#dataset)<br>
# [Importação de dados](#import2)<br>
# [Limpeza dos dados de treino](#limpeza)<br>
# [Tokenização e sequênciação](#token)<br>
# [Criação e parametrização da rede](#criacao)<br>
# [Separação de dados em treino e teste](#split)<br>
# [Efectivação do treino](#treino)<br>
# [Testes](#testes)<br>
# [Avaliação](#avaliacao)<br>
# [Conclusão](#conclusao)
# [Referências](#referencias)

# <p>&nbsp;</p>
# <p>&nbsp;</p>
# <p>&nbsp;</p>
# 
# <a id="super"></a>
# # Análise sentimental supervisionada (LSTM)
# <p>&nbsp;</p>
# 
# Agora que temos informação de como se comporta um dos métodos de classificação, em que reconhecemos pontos fortes e pontos fracos vamos abordar o mesmo <i>dataset</i> usando uma rede neuronal recorrente. As redes [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory), Long short-term memory são redes que consomem sequências de dados, com memória mas com a capacidade de esquecer. Acreditamos que esta característica poderá superar os pontos fracos do classificado Naive Bayes.
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
#     pip install tensorflow
#     
#     
# <div class="alert alert-block alert-info">Para quem tiver uma gráfica das mais recentes gerações sugere-se a instalação do "tensorflow-gpu"</div>
# <div class="alert alert-block alert-info">NOTA: Se não tiveres premissões para instalar, executa o Prompt como Administrador. Se estiveres da drive C: e os notebooks estiverem na D:, assim que abres o Prompt deves escrever D: para mudar de drive.</div>
# 
# No nosso caso vamos usar um contentor dinâmico que permite a partilha de notebooks com edição, [mybinder.org](https://mybinder.org).

# <a id="import"></a>
# ### A importação dos módulos necessários

# In[ ]:


# algebra linear
import numpy as np 
# processamento de dados e I/O ficheiros 
import pandas as pd 
# Biblioteca de vectorização
from sklearn.feature_extraction.text import CountVectorizer
# Biblioteca de tokenização
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re


# <a id="dataset"></a>
# ### O Dataset
# 
# Para um <i>benchmark</i> fazer sentido devemos usar os mesmo dados. Se bem que não controlamos exactamente que parcelas são usadas para treino e teste acreditamos poder aferir se há melhorias entre um método de classificação e o outro. 
# 

# <a id="import2"></a>
# ### Importação dos dados
# 
# Os nossos <i>dataset</i> são CSVs que contém diversa informação, como o nosso objectivo é fazer análise sentimental precisamos apenas das colunas com o texto do tweet e a classificação atribuida por utilizadores. 

# In[ ]:


#data = pd.read_csv('../_datasets/GOP_REL_ONLY.csv', encoding = "ISO-8859-1")
data = pd.read_csv('../_datasets/1377191648_sentiment_nuclear_power.csv', encoding = "ISO-8859-1")
# Keeping only the neccessary columns
data = data[['text','sentiment']]


# <a id="limpeza"></a>
# ### Limpeza dos dados
# 
# Seguidamente vamos padronizar os dados, remover alguma informação desnecessária para garantir que ficamos apenas com palavras e, como fizémos com o classificador anterior, vamos excluir os <i>tweets</i> neutros pois pretendemos obter diferenciação.
# 

# In[ ]:


data = data[data.sentiment != "Neutral"]
data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

print(data[ data['sentiment'] == 'Positive'].size)
print(data[ data['sentiment'] == 'Negative'].size)

for idx,row in data.iterrows():
    row[0] = row[0].replace('rt',' ')


# <a id="token"></a>
# ### Tokenização e sequênciação
# 
# Then, I define the number of max features as 2000 and use Tokenizer to vectorize and convert text into Sequences so the Network can deal with it as input.

# In[ ]:


max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)


# <a id="criacao"></a>
# ### Criação e parametrização da rede
# 
# Agora vamos criar a rede the LSTM. 
# 
# A configuração de parâmetros como **embed_dim**, **lstm_out**, **batch_size**, **droupout** e **rate** requerem experimentação até encontrar os melhores resultados. 
# 
# Vamos usar a função "[softmax](https://en.wikipedia.org/wiki/Softmax_function)" para activação do modelo pois num rede de [entropia cruzada](https://en.wikipedia.org/wiki/Cross_entropy) essa é a abordagem correcta.
# 
# <div class="alert alert-block alert-info">A função **softmax** comprime um vector de qualquer dimensão com quaisquer números reais para o intervalo [0,1]</div>
# 
# <div class="alert alert-block alert-info">**Entropia cruzada** entre duas distruições permite estimar o tamanho de informação necessária para identificar a que distribuição pertence um elemento</div>

# In[ ]:


embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(rate=0.2))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


# <a id="split"></a>
# ### Separação de dados em treino e teste
# 
# Poderemos "brincar" com a proporção para perceber como influencia os resultados.
# 
# **random_state** define a aleatoridade com que a rede se lembra e esquece.

# In[ ]:


Y = pd.get_dummies(data['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# <a id="treino"></a>
# ### Efectivação do treino
# 
# Agora resta-nos efectuar o treino. Idealmente deveríamos de usar um elevado número de <i>epochs</i> mas em nome da fluidez da explicação vamos restringir-nos a 9.

# In[ ]:


batch_size = 32
model.fit(X_train, Y_train, epochs = 9, batch_size=batch_size, verbose = 2)


# <a id="teste"></a>
# ## Agora um teste
# 
# Vamos pegar em alguns dados, testá-los e medir verificar um par de indicadores de performance:

# In[ ]:


validation_size = 1500

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)

print("Indicadores de performance:")
print("score: %.2f%%" % (score*100))
print("acc: %.2f%%" % (acc*100))


# <a id="avaliacao"></a>
# ### Avaliação de resultados
# 
# Vamos medir o número de acertos e reflectir sobre eles e posteriormente compará-los com o outro classificador.

# In[ ]:


pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
for x in range(len(X_validate)):
    
    result = model.predict(X_validate[x].reshape(1,X_test.shape[1]),batch_size=1,verbose = 2)[0]
   
    if np.argmax(result) == np.argmax(Y_validate[x]):
        if np.argmax(Y_validate[x]) == 0:
            neg_correct += 1
        else:
            pos_correct += 1
       
    if np.argmax(Y_validate[x]) == 0:
        neg_cnt += 1
    else:
        pos_cnt += 1

print('[Negative]: %0.2f%%'  % (neg_correct/neg_cnt*100))       
print('[Positive]: %0.2f%%'  % (pos_correct/pos_cnt*100))


# <a id="conclusao"></a>
# ## Conclusão
# 
# Pelos resultados obtidos torna-se claro que ambos os métodos são muito bons a encontraar tweets negativos mas no momento da decisão se são positivos algo falha. Podemos especular que são os próprios <i>datasets</i> que têm muito mais <i>tweets</> negativos que positivos. Para isso temos outro <i>dataset</i>. Havendo tempo podemos efectuar alguma experiências.
# 
# E sobre análise de sentimentos? Estes exercícios mais técnicos pretendem demonstrar que os resultados de uma análise não dependem exclusivamente do peso ou conotação das palavras mas também dos métodos e parametros usados nos classificadores, da precisão da categorização dos dados de treino, da vizinhança das palavras etc. 
# 
# Um dos métodos que gostaríamos de ter explorado baseia-se em florestas aleatórias de decisão, que afectam consideravelmente o a conotação de uma palavra com as suas vizinhas (como podémos testar nos 2 primeiros exercícios). No entanto, apesar de termos feito investigação e experiências, esse <i>notebook</i> seria já muito pesado e faria-nos exceder os objectivos da unidade curricular.

# [<div align="right" class="alert alert-block"><img src="../_images/top.png" width="20"></div>](#topo)

# ____
# <a id="referencias"></a>
# 
# # Referências
# 
# http://billchambers.me/tutorials/2015/01/14/python-nlp-cheatsheet-nltk-scikit-learn.html<br>
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/SpatialDropout1D<br>
# https://en.wikipedia.org/wiki/Cross_entropy<br>
# https://en.wikipedia.org/wiki/Softmax_function
