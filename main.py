import os
import pandas as pd
from preprocess import trainDoc2Vec, vectorizeSentence

"""
target = real(0) or fake(1)
"""

root = os.path.dirname(os.path.abspath(__file__))

def createVectorizedDataset(filename):
    df = pd.read_csv(os.path.join(root, 'data', filename))
    df = df.head(50)

    df = df.drop(columns=['tweet_ids', 'id'])
    df['target'] = 1
    print(df.head(10))

    # train model on all of the sentences 
    modelPath = os.path.join(root, 'data', 'all_titles.bin')
    title_string = ' '.join(df['title'])
    trainDoc2Vec(title_string, modelPath)

    # Vectorize each sentence
    for index, row in df.iterrows():
        df.at[index, 'title'] = vectorizeSentence(row['title'], modelPath)

    # save the df 
    newName = filename + '_vectorized.csv'
    df.to_csv(os.path.join(root, 'data', newName), index=False)


createVectorizedDataset('politifact_fake.csv')
# createVectorizedDataset('politifact_real.csv')

# combine both datasets