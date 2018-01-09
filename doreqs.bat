cls
pip install pipreqs
rem jupyter nbconvert --to=python "movies reviews\*.ipynb"
jupyter nbconvert --to=python "MLvsLSTM\*.ipynb"
jupyter nbconvert --to=python "Textblob\*.ipynb"
pipreqs . --ignore Mreviews --force 