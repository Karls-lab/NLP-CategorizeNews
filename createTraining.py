import os
import pandas as pd
import sys
from preprocess import trainDoc2Vec, vectorizeSentence

# Declare the path for models and root
root = os.path.dirname(os.path.abspath(__file__))
fakeNewsModel = os.path.join(root, 'models', 'fake_titles.bin')
realNewsModel = os.path.join(root, 'models', 'real_titles.bin')


def createDoc2Vec(filename, modelPath):
    df = pd.read_csv(os.path.join(root, 'data', filename))
    df = df.drop(columns=['tweet_ids', 'id'])
    allSentences = ' '.join(df['title'])
    trainDoc2Vec(allSentences, modelPath)


def createVectorizedDataset(filename, target, modelPath):
    df = pd.read_csv(os.path.join(root, 'data', filename))
    df = df.drop(columns=['news_url', 'tweet_ids', 'id'])
    df['target'] = target

    # Vectorize each sentence
    for index, row in df.iterrows():
        featureVector = pd.Series(vectorizeSentence(row['title'], modelPath))
        # Feature vector is 10 elements long and represents the sentence
        for i in range(len(featureVector)):
            df.at[index, 'feature{}'.format(i)] = featureVector[i]

    # Save the new dataset and drop the sentence
    newName = filename + f'_vectorized.csv'
    df = df.drop(columns=['title'])
    df.to_csv(os.path.join(root, 'data', newName), index=False)


# Create and train doc2vec models on fake and real news sentences
createDoc2Vec('politifact_fake.csv', fakeNewsModel)
createDoc2Vec('politifact_real.csv', realNewsModel)

# Turn the sentences into vectors and save the new dataset
# features are the 10 elements of the sentence vector
modelFake = os.path.join(root, 'models', 'fake_titles.bin')
modelReal = os.path.join(root, 'models', 'real_titles.bin')
createVectorizedDataset('politifact_fake.csv', 1, modelFake)
createVectorizedDataset('politifact_real.csv', 0, modelReal)