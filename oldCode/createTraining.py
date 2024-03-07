import os
import pandas as pd
import sys
from oldCode.preprocess import trainDoc2Vec, vectorizeSentence
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords

# Declare the path for models and root
root = os.path.dirname(os.path.abspath(__file__))

def createDoc2Vec(df, textCol, category, modelPath):
    df = df[df[category] == True]
    allSentences = ' '.join(df[textCol])
    print(f'Training model for category: {category}')
    print(f'len of allSentences: {len(allSentences)}')
    if not os.path.exists(modelPath):
        trainDoc2Vec(allSentences, modelPath)
    else: print("Model already exists, skipping training\n")


def createVectorizedDataset(df, textCol, filename, modelPath):
    newName = filename + f'_vectorized.csv'
    print(f'Creating vectorized dataset for {newName}...')
    if os.path.exists(os.path.join(root, 'Doc2Vec_Data', newName)):
        print(f'{filename} already exists, skipping creation\n')
        return
    # Vectorize each sentence
    for index, row in df.iterrows():
        featureVector = pd.Series(vectorizeSentence(row[textCol], modelPath))
        # Feature vector is 10 elements long and represents the sentence
        for i in range(len(featureVector)):
            df.at[index, 'feature{}'.format(i)] = featureVector[i]

    # Save the new dataset and drop the sentence
    df = df.drop(columns=[textCol])
    folder = os.path.join(root, 'Doc2Vec_Data')
    if not os.path.exists(folder):
        os.makedirs(folder)
    fileName = os.path.join(folder, newName)
    if not os.path.exists(fileName):
        df.to_csv(fileName, index=False)
    else: print("File already exists, skipping creation\n")


def vectorizeData(df, X_train, X_test):
    cv = CountVectorizer(stop_words=stopwords.words('english'))
    X_train_counts = cv.fit_transform(X_train["title"])
    X_test_counts = cv.transform(X_test["title"])
    return X_train_counts, X_test_counts, cv


"""
Read in the data, we are only interested in headline and category. 
One hot encode the categories
"""
df = pd.read_json(os.path.join(root, 'data', 'News_Category_Dataset_v3.json'), lines=True)
df = df[['headline', 'category']]
# print(df['category'].unique())
# get only subset of categories
df = df[df['category'].isin(['POLITICS', 'ENTERTAINMENT', 'TECH'])]
df = pd.get_dummies(df, columns=['category'])
print(df.columns)




sys.exit()

"""Turn the sentences into vectors and save the new dataset
   features are the 10 elements of the sentence vector"""
for category in df.columns[4:7]:
    modelPath = os.path.join(root, 'models', f'{category}.bin')
    # uses only the subset, not the entire df
    category_df = df[df[category] == True].reset_index()
    category_df = category_df[['index', 'headline'] + list(df.columns[4:7])]
    print(f'columsns of dataframe: {category_df.columns}')
    createDoc2Vec(category_df, 'headline', category, modelPath)
    createVectorizedDataset(category_df, 'headline', f'{category}.csv', modelPath)


"""
for each dataset in Doc2Vec_Data, merge into one dataset and save
"""
allData = pd.DataFrame()
for category in df.columns[4:7]:
    filename = f'{category}.csv_vectorized.csv'
    df = pd.read_csv(os.path.join(root, 'Doc2Vec_Data', filename))
    allData = pd.concat([allData, df], ignore_index=True)
allData.to_csv(os.path.join(root, 'Doc2Vec_Data', 'allData.csv'), index=False)

