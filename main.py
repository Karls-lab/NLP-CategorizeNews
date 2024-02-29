from preprocess import splitTrainingData
from model import NLTK_Binary_Classifier
import os 
import pandas as pd

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    dfReal = pd.read_csv(os.path.join(root, 'data', 'politifact_real.csv_vectorized.csv'))
    dfFake = pd.read_csv(os.path.join(root, 'data', 'politifact_fake.csv_vectorized.csv'))
    df = pd.concat([dfReal, dfFake])
    columns = df.columns
    X_train, X_test, y_train, y_test = splitTrainingData(df, columns[1:], 'target')
    model = NLTK_Binary_Classifier()
    model.compile()

    print(X_train.head())
    print(y_train.head())
    history = model.fit(X_train, y_train, epochs=50, batch_size=64)
    model.reset_weights()
    



if __name__ == '__main__':
    main()
