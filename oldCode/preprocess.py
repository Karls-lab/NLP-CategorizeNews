import nltk
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


""" Tokenization, Stopwords removal and Lemmatization """
def lemmatizeSentence(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word, pos=wordnet.VERB) for word in filtered_tokens]
    return lemmatized_tokens


""" Train the Doc2Vec model on the given text and save it to the given path"""
def trainDoc2Vec(text, modelPath):
    print(f'Training model on text: {text[0:200]}')
    print(f'creating model saved to: {modelPath}\n')
    tokens = lemmatizeSentence(text)
    tagged_data = [TaggedDocument(words=words, tags=[str(i)]) for i, words in enumerate(tokens)]
    model = Doc2Vec(vector_size=10, window=50, min_count=1, epochs=32)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)    
    model.save(modelPath)


def vectorizeSentence(text, modelPath):
    tokens = lemmatizeSentence(text)
    model = Doc2Vec.load(modelPath)
    sentence_embedding = model.infer_vector(tokens)
    return sentence_embedding


def splitTrainingData(df, featureCols, targetCols, random=False):
    state = 42 if random else None
    X = df[featureCols]
    y = df[targetCols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=state)
    return X_train, X_test, y_train, y_test

