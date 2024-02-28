import nltk
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

def lemmatizeSentence(text):
    # Tokenization, Stopwords removal and Lemmatization
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word, pos=wordnet.VERB) for word in filtered_tokens]
    return lemmatized_tokens


def trainDoc2Vec(text, savePath):
    print(f'save path: {savePath}')
    tokens = lemmatizeSentence(text)
    """ Compute Doc2Vec Embedding """
    tagged_data = [TaggedDocument(words=words, tags=[str(i)]) for i, words in enumerate(tokens)]
    model = Doc2Vec(vector_size=10, window=5, min_count=1, epochs=20)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    sentence_embedding = model.infer_vector(tokens)

    model.save(savePath)


def vectorizeSentence(text, modelPath):
    tokens = lemmatizeSentence(text)
    model = Doc2Vec.load(modelPath)
    sentence_embedding = model.infer_vector(tokens)
    return sentence_embedding



    # tagged_data = [TaggedDocument(words=lemmatized_tokens, tags=[str(index)])]