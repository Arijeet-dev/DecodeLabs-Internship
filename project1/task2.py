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

print("Starting Data Cleaning & Feature Engineering...\n")
print("Missing values BEFORE cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])
df['age'] = df.groupby(['pclass', 'sex'])['age'].transform(lambda x: x.fillna(x.median()))
embarked_mode = df['embarked'].mode()[0]
df['embarked'] = df['embarked'].fillna(embarked_mode)
if 'embark_town' in df.columns:
    embark_town_mode = df['embark_town'].mode()[0]
    df['embark_town'] = df['embark_town'].fillna(embark_town_mode)
if 'deck' in df.columns:
    df['deck'] = df['deck'].astype(str).replace('nan', 'Unknown')
    df['deck'] = df['deck'].astype('category')
if 'cabin' in df.columns:
    df['cabin'] = df['cabin'].fillna('Unknown')
print("Missing values AFTER cleaning:")
print(df.isnull().sum())

# Explaining why we don't drop duplicates
duplicate_count = df.duplicated().sum()
print(f"Number of duplicate rows found based on features: {duplicate_count}")
print("Note: These rows represent distinct passengers sharing identical features, not duplicate entries. We retain them to preserve data integrity.")

categorical_cols = ['survived', 'pclass', 'sex', 'embarked']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')
df['age'] = df['age'].round().astype(int)
print("Data types after conversion:")
print(df.dtypes[['survived', 'pclass', 'sex', 'age', 'embarked']])
Q1 = df['fare'].quantile(0.25)
Q3 = df['fare'].quantile(0.75)
IQR = Q3 - Q1
upper_fence = Q3 + 1.5 * IQR
lower_fence = max(0, Q1 - 1.5 * IQR)
print(f"Fare Outlier Threshold (Upper Fence): ${upper_fence:.2f}")
outliers_count = (df['fare'] > upper_fence).sum()
print(f"Number of outlier fares above threshold: {outliers_count}")
df['fare_capped'] = np.where(df['fare'] > upper_fence, upper_fence, df['fare'])
print("Outliers handled: Created 'fare_capped' column with values capped at upper fence limit.")
df['FamilySize'] = df['sibsp'] + df['parch'] + 1
print("1. Created 'FamilySize' feature (sibsp + parch + 1)")
df['IsAlone'] = np.where(df['FamilySize'] == 1, 1, 0)
df['IsAlone'] = df['IsAlone'].astype('category')
print("2. Created 'IsAlone' feature (1 if FamilySize == 1, else 0)")
df['FarePerPerson'] = df['fare'] / df['FamilySize']
print("3. Created 'FarePerPerson' feature (fare / FamilySize)")
print("\nPreview of the final preprocessed dataset:")
preview_cols = ['sex', 'age', 'FamilySize', 'IsAlone', 'fare', 'fare_capped', 'FarePerPerson']
print(df[preview_cols].head())

