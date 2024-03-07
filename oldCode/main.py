from oldCode.preprocess import splitTrainingData
from oldCode.model import NLTK_Classifier
import os 
import pandas as pd

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(root, 'Doc2Vec_Data', 'allData.csv'))

    feature_columns = df.filter(like='feature').columns.tolist()
    category_columns = df.filter(like='category').columns.tolist()
    X_train, X_test, y_train, y_test = splitTrainingData(df, feature_columns, category_columns)

    X_train= X_train.values.reshape(-1, 1, 10)
    X_test= X_test.values.reshape(-1, 1, 10)

    model = NLTK_Classifier(numClasses=3)
    model.compile()

    # print(X_train.head())
    # print(y_train.head())
    history = model.fit(X_train, y_train, epochs=32, batch_size=64)
    # model.reset_weights()
    


if __name__ == '__main__':
    main()
