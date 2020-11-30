# LSI

## Preprocessing

The preprocessing of the data is implemented in the
[preprocessing.py](./src/preprocessing.py) script.

### NLTK

For the lemmatization step, and for splitting the text into words and sentences,
we use the NLTK library. This library can be installed using `pip install nltk`.
This, however, does not install all of the required files. In order to run the
[preprocessing.py](./src/preprocessing.py) script, you first have to run
[setup.py](./nltk/setup.py). This will install all of the data, required for our
code, in the `nltk` directory.

