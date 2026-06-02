import numpy as np
import pandas as pd
import seaborn as sns
try:
    df = sns.load_dataset('titanic')
    print("Dataset successfully loaded from Seaborn.")
except Exception as e:
    print(f"Seaborn load failed: {e}. Attempting load from direct pandas URL...")
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    rename_map = {
        'Survived': 'survived',
        'Pclass': 'pclass',
        'Sex': 'sex',
        'Age': 'age',
        'SibSp': 'sibsp',
        'Parch': 'parch',
        'Fare': 'fare',
        'Embarked': 'embarked'
    }
    df = df.rename(columns=rename_map)
    if 'pclass' in df.columns:
        df['class'] = df['pclass'].map({1: 'First', 2: 'Second', 3: 'Third'}).astype('category')
    if 'sex' in df.columns:
        df['who'] = np.where(df['age'] < 16, 'child', np.where(df['sex'] == 'female', 'woman', 'man'))
        df['adult_male'] = (df['sex'] == 'male') & (df['age'] >= 16)
    if 'Cabin' in df.columns:
        df['deck'] = df['Cabin'].dropna().str[0].astype('category')
    if 'embarked' in df.columns:
        df['embark_town'] = df['embarked'].map({'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'})
    if 'survived' in df.columns:
        df['alive'] = np.where(df['survived'] == 1, 'yes', 'no')
    if 'sibsp' in df.columns and 'parch' in df.columns:
        df['alone'] = (df['sibsp'] + df['parch']) == 0
    print("Dataset successfully loaded and normalized from GitHub URL.")


print(f"Dataset Shape (Rows, Columns): {df.shape}\n")
print("DataFrame Information:")
df.info()
print("First 5 rows of the dataset:")
print(df.head(5))
print("Last 5 rows of the dataset:")
print(df.tail(5))
print("Summary Statistics (Numerical Columns):")
print(df.describe())
print("Summary Statistics (Categorical/Object Columns):")
try:
    print(df.describe(include=['object', 'category', 'bool']))
except Exception:
    print(df.describe(include='all'))
