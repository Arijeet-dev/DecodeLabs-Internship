import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
script_dir = os.path.dirname(os.path.abspath(__file__))
viz_dir = os.path.join(script_dir, 'visualizations')
os.makedirs(viz_dir, exist_ok=True)
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

df['age'] = df.groupby(['pclass', 'sex'])['age'].transform(lambda x: x.fillna(x.median()))
embarked_mode = df['embarked'].mode()[0]
df['embarked'] = df['embarked'].fillna(embarked_mode)
if 'embark_town' in df.columns:
    df['embark_town'] = df['embark_town'].fillna(df['embark_town'].mode()[0])
if 'deck' in df.columns:
    df['deck'] = df['deck'].astype(str).replace('nan', 'Unknown').astype('category')

# Duplicates are retained to preserve data integrity and distinct passenger records
categorical_cols = ['survived', 'pclass', 'sex', 'embarked']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')
df['age'] = df['age'].round().astype(int)
Q1 = df['fare'].quantile(0.25)
Q3 = df['fare'].quantile(0.75)
IQR = Q3 - Q1
upper_fence = Q3 + 1.5 * IQR
df['fare_capped'] = np.where(df['fare'] > upper_fence, upper_fence, df['fare'])
df['FamilySize'] = df['sibsp'] + df['parch'] + 1
df['IsAlone'] = np.where(df['FamilySize'] == 1, 1, 0)
df['FarePerPerson'] = df['fare'] / df['FamilySize']
age_bins = [0, 12, 18, 35, 60, 100]
age_labels = ['Child (0-12)', 'Teenager (13-18)', 'Young Adult (19-35)', 'Adult (36-60)', 'Senior (60+)']
df['AgeGroup'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, include_lowest=True)
df['survived_numeric'] = df['survived'].astype(int)
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    'figure.titlesize': 18,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.facecolor': '#f8f9fa'
})
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x='sex', y='survived_numeric', hue='sex', palette='muted', legend=False, errorbar=None, edgecolor='0.2')
plt.title('Survival Rate by Gender', pad=15, fontweight='bold', color='#2b2d42')
plt.xlabel('Gender')
plt.ylabel('Survival Rate')
plt.ylim(0, 1)
for p in plt.gca().patches:
    plt.gca().annotate(f"{p.get_height():.1%}", (p.get_x() + p.get_width() / 2., p.get_height() + 0.02),
                ha='center', va='center', fontsize=11, color='black', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'plot1_survival_gender.png'), dpi=150)
plt.close()
print("Saved: plot1_survival_gender.png")
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x='pclass', y='survived_numeric', hue='pclass', palette='viridis', legend=False, errorbar=None, edgecolor='0.2')
plt.title('Survival Rate by Passenger Class', pad=15, fontweight='bold', color='#2b2d42')
plt.xlabel('Passenger Ticket Class')
plt.ylabel('Survival Rate')
plt.ylim(0, 1)
for p in plt.gca().patches:
    plt.gca().annotate(f"{p.get_height():.1%}", (p.get_x() + p.get_width() / 2., p.get_height() + 0.02),
                ha='center', va='center', fontsize=11, color='black', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'plot2_survival_class.png'), dpi=150)
plt.close()
print("Saved: plot2_survival_class.png")
plt.figure(figsize=(9, 5))
sns.barplot(data=df, x='AgeGroup', y='survived_numeric', hue='AgeGroup', palette='coolwarm', legend=False, errorbar=None, edgecolor='0.2')
plt.title('Survival Rate by Age Group', pad=15, fontweight='bold', color='#2b2d42')
plt.xlabel('Age Category')
plt.ylabel('Survival Rate')
plt.ylim(0, 1)
for p in plt.gca().patches:
    plt.gca().annotate(f"{p.get_height():.1%}", (p.get_x() + p.get_width() / 2., p.get_height() + 0.02),
                ha='center', va='center', fontsize=10, color='black', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'plot3_survival_agegroup.png'), dpi=150)
plt.close()
print("Saved: plot3_survival_agegroup.png")
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='age', kde=True, color='#457b9d', edgecolor='w', bins=25)
plt.title('Passenger Age Distribution (Cleaned)', pad=15, fontweight='bold', color='#2b2d42')
plt.xlabel('Age (Years)')
plt.ylabel('Passenger Count')
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'plot4_age_distribution.png'), dpi=150)
plt.close()
print("Saved: plot4_age_distribution.png")
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(data=df, x='fare', kde=True, ax=axes[0], color='#e63946', bins=30)
axes[0].set_title('Original Fare Distribution (Extreme Outliers)', fontweight='bold')
axes[0].set_xlabel('Fare ($)')
axes[0].set_ylabel('Count')
sns.histplot(data=df, x='fare_capped', kde=True, ax=axes[1], color='#2a9d8f', bins=30)
axes[1].set_title('Capped Fare Distribution (IQR Capped at $65.63)', fontweight='bold')
axes[1].set_xlabel('Capped Fare ($)')
axes[1].set_ylabel('Count')
plt.suptitle('Comparison of Ticket Fare Distributions', y=1.02, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'plot5_fare_distribution.png'), dpi=150)
plt.close()
print("Saved: plot5_fare_distribution.png")
plt.figure(figsize=(9, 7))
df['sex_numeric'] = np.where(df['sex'] == 'female', 1, 0)
df['IsAlone_numeric'] = df['IsAlone'].astype(int)
numerical_cols = ['survived_numeric', 'pclass', 'sex_numeric', 'age', 'fare_capped', 'FamilySize', 'IsAlone_numeric']
corr_matrix = df[numerical_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation Matrix Heatmap', pad=20, fontweight='bold', color='#2b2d42')
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'plot6_correlation_heatmap.png'), dpi=150)
plt.close()
print("Saved: plot6_correlation_heatmap.png")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.boxplot(data=df, x='pclass', y='age', ax=axes[0], hue='pclass', palette='pastel', legend=False, width=0.5)
axes[0].set_title('Age Distribution by Passenger Class', fontweight='bold')
axes[0].set_xlabel('Passenger Class')
axes[0].set_ylabel('Age')
sns.boxplot(data=df, x='pclass', y='fare_capped', ax=axes[1], hue='pclass', palette='pastel', legend=False, width=0.5)
axes[1].set_title('Capped Fare Distribution by Passenger Class', fontweight='bold')
axes[1].set_xlabel('Passenger Class')
axes[1].set_ylabel('Fare ($)')
plt.suptitle('Demographics and Fares Across Passenger Classes', y=1.02, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'plot7_boxplots.png'), dpi=150)
plt.close()
print("Saved: plot7_boxplots.png")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.countplot(data=df, x='sex', ax=axes[0, 0], hue='sex', palette='Set2', legend=False, edgecolor='0.2')
axes[0, 0].set_title('Passenger Count by Gender', fontweight='bold')
axes[0, 0].set_xlabel('Gender')
axes[0, 0].set_ylabel('Count')
sns.countplot(data=df, x='pclass', ax=axes[0, 1], hue='pclass', palette='Set2', legend=False, edgecolor='0.2')
axes[0, 1].set_title('Passenger Count by Class', fontweight='bold')
axes[0, 1].set_xlabel('Class')
axes[0, 1].set_ylabel('Count')
sns.countplot(data=df, x='embark_town', ax=axes[1, 0], hue='embark_town', palette='Set2', legend=False, edgecolor='0.2')
axes[1, 0].set_title('Passenger Count by Embarkation Town', fontweight='bold')
axes[1, 0].set_xlabel('Embarkation Town')
axes[1, 0].set_ylabel('Count')
sns.countplot(data=df, x='IsAlone', ax=axes[1, 1], hue='IsAlone', palette='Set2', legend=False, edgecolor='0.2')
axes[1, 1].set_title('Passenger Count by Traveling Status (Alone = 1)', fontweight='bold')
axes[1, 1].set_xlabel('Is Alone (1 = Yes, 0 = No)')
axes[1, 1].set_ylabel('Count')
plt.suptitle('Distribution of Categorical Variable Frequencies', y=1.02, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'plot8_categorical_counts.png'), dpi=150)
plt.close()
print("Saved: plot8_categorical_counts.png")
plt.figure(figsize=(9, 6))
sns.scatterplot(data=df, x='age', y='fare_capped', hue='survived', style='survived', palette='Set1', alpha=0.7, s=70)
plt.title('Scatter Plot: Age vs Capped Fare (Colored by Survival)', pad=15, fontweight='bold', color='#2b2d42')
plt.xlabel('Age (Years)')
plt.ylabel('Capped Fare ($)')
plt.legend(title='Survived (0 = No, 1 = Yes)', loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'plot9_scatter_age_fare.png'), dpi=150)
plt.close()
print("Saved: plot9_scatter_age_fare.png")

