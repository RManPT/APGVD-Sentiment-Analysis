
# coding: utf-8

# # Natural language processing
# 

# Primeiro que tudo temos de fazer a importação de módulos e bibliotecas necessárias.
# 
# Vamos começar por "os" para termos acesso a funções da linha de comandos:
# 

# In[6]:


import os


# Depois vamos adicionar um módulo de "machine learning" de rápida aprendizagem baseado em florestas de decisão:

# In[7]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


# De seguida vamos importar uma classe que nos vai ajudar a limpar os nossos dados:

# In[8]:



from KaggleWord2VecUtility import KaggleWord2VecUtility


# Precisamos também de ter acesso aos ficheiro CSV e para isso usamos o Pandas:

# In[9]:


import pandas as pd


# Precisamos do 'numpy' para lidar com arrays multidimensionais

# In[10]:


import numpy as np


# E o NLTK, Natural Language Tool Kit, que nos ajudará a remover palavras desnecessárias do nosso texto.

# In[11]:


import nltk


# ## Primeiro passo
# 
# Ler os dados de um ficheiro, os dados de treino e os dados de teste. Seguidamente imprimimos a primeira review.
# 

# In[13]:


if __name__ == '__main__':
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0,                     delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t",                    quoting=3 )

    print("The first review is:")
    print(train["review"][0])

    raw_input("Press Enter to continue...")


# ## Segundo passo
# 
# Obter os dados de treino

# In[ ]:


print('Download text data sets. If you already have NLTK datasets downloaded, just close the Python download window...')
nltk.download()


# ## Terceiro passo
# 
# Limpar os dados de tags de HTML, caracteres não alfabuméricos e 'Stop words':

# In[ ]:


# Initialize an empty list to hold the clean reviews
clean_train_reviews = []
print("Cleaning and parsing the training set movie reviews...\n")


# Agora vamos iterar por todo o conjunto de dados de treino e garantir que a sua limpeza:

# In[5]:


# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list

print("Cleaning and parsing the training set movie reviews...\n")
for i in xrange( 0, len(train["review"])):
    clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], True)))


# ### Criação do dicionário
# 
# Agora vamos criar o nosso "Bag of Words". O modelo "Bag of Words" é uma simples representação numérica do conteúdo do nosso texto que é fácil de classificar. Basicamente associa a cada palavra a sua frequência e cria um dicionário. A este processo chama-se tokenização em processamento de linguagem natural.
# 
# Vamos usar o objecto CountVectorizer para o gerar. Vamos parametrizá-lo com         
# um máximo de 5000 palavras para simplificar.

# In[ ]:


# ****** Create a bag of words from the training set
#
print("Creating the bag of words...\n")


# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
vectorizer = CountVectorizer(analyzer = "word",   
                         tokenizer = None,    
                         preprocessor = None, 
                         stop_words = None,   
                         max_features = 5000)


# ### Fitting ao modelo
# 
# Criação dos vectores de 'features'

# In[7]:


# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of
# strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an
# array
np.asarray(train_data_features)


# # Passo quatro
# 
# Criação de um classificador baseado num floresta de decisão aleatória com 100 árvores. Uma árvore de decisão é um grafo que modelam a probabilidade de certos eventos (resultados).
# 
# Ao usar várias árvores podemos fazer depender a precisão de uma palavra com as das que a rodeiam.

# In[ ]:


# ******* Train a random forest using the bag of words
#
print("Training the random forest (this may take a while)...")

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100)
# Fit the forest to the training set, using the bag of words as
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, train["sentiment"] )
# Create an empty list and append the clean reviews one by one
clean_test_reviews = []


# # Passo cinco
# 
# Formatação dos dados de teste.
# 
# Vamos limpar as reviews e criar um 'Bag of words'

# In[ ]:


print("Cleaning and parsing the test set movie reviews...\n")
for i in xrange(0,len(test["review"])):
    clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], True)))


# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
np.asarray(test_data_features)


# In[12]:


# Use the random forest to make sentiment label predictions
print("Predicting test labels...\n")
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'Bag_of_Words_model.csv'), index=False, quoting=3)
print("Wrote results to Bag_of_Words_model.csv")

