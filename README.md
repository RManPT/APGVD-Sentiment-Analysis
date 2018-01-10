First steps to explore the project on your CLI:

- Go to folder _resources and run "pip install wordcloud-1.3.3-cp36-cp36m-win_amd64"

- Back on root, run "pip install -r requirements.txt"

- Start jupyter Network and enjoy



If you encounter any problems installing the required dependencies we sugest you to Fork the project and use https://mybinder.org/ so you can edit the notebook on a dynamic docker image. All the requirements are automatically installed and all will work just fine.
DISAVANTAGE: If you have a super fast computer you may think the neural networks running on mybinder are sloooww.



If you want to edit the notebooks adding imports you may make sure they are installed smoothly by running "doreqs".
This batch file will detect the requirements for the notebooks and generate a new requirements file.
OBS: You must add Tensorflow manually "tensorflow==1.4.0"

Then just commit the changes into your git repo and restart you mybinder image... voilá!



Notebook list:


# First steps into sentiment analisys
\TextBlob\
  Ex1 - Sentiment Analysis in 3 Lines
  Ex2 - Twitter Sentiment Analysis
  
# Messing around with classifiers
\MLvsLSTM
  Ex3 - Naivebayes
  Ex4 - Lstm
  
  
 Resources:
 
 \_images         - images needed for notebooks
 \_datasets       - sourced from crowdflower to train and test the  classifiers
 \_dependencies   - WordCloud module
