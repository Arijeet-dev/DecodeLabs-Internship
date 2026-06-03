

---

##  Project Structure

```text
DecodeLabs-Internship/
├── project1/
│   ├── task1.py                 # Task 1: Data Loading & Initial Exploration
│   ├── task2.py                 # Task 2: Data Cleaning & Feature Engineering
│   ├── task3.py                 # Task 3: Statistical Analysis & Business Insights
│   ├── task4.py                 # Task 4: Comprehensive Data Visualization
│   └── visualizations/          # Generated Data Visualizations
│       ├── plot1_survival_gender.png
│       ├── plot2_survival_class.png
│       ├── plot3_survival_agegroup.png
│       ├── plot4_age_distribution.png
│       ├── plot5_fare_distribution.png
│       ├── plot6_correlation_heatmap.png
│       ├── plot7_boxplots.png
│       ├── plot8_categorical_counts.png
│       └── plot9_scatter_age_fare.png
├── LICENSE                      # Project License
└── README.md                    # Project Documentation (This file)
```

---

##  Tasks Breakdown

###  Task 1: Data Loading & Initial Exploration
- **File**: [`project1/task1.py`](file:///c:/Users/KIIT/OneDrive/Desktop/DecodeLabs-Internship/project1/task1.py)
- **Objective**: Establish a robust data loading pipeline and perform initial high-level exploratory analysis.
- **Key Features**:
  - **Dual-Source Loading**: Attempts to load the Titanic dataset directly from Seaborn's built-in datasets, with an automatic fallback to a raw CSV URL on GitHub if the local load fails.
  - **Schema Normalization**: Normalizes and maps column names in the fallback path to ensure perfect schema consistency with the standard Seaborn dataset.
  - **Exploration Output**: Displays basic structural properties including dataset shape, columns info, first and last 5 rows, and separate summary statistics for numerical vs. categorical variables.

###  Task 2: Data Cleaning & Feature Engineering
- **File**: [`project1/task2.py`](file:///c:/Users/KIIT/OneDrive/Desktop/DecodeLabs-Internship/project1/task2.py)
- **Objective**: Handle missing values, optimize data types, cap outliers, and engineer new domain-specific features.
- **Key Features**:
  - **Imputation**: Missing age values are imputed using the group median based on passenger class and gender (`pclass` + `sex`). Embarked details are filled with their mode, and missing deck/cabin values are categorized as `'Unknown'`.
  - **Data Integrity**: Decides to retain duplicate records, justifying that these represent distinct passengers sharing identical features rather than duplicate entries.
  - **Outlier Capping**: Caps extreme fares at the IQR upper fence ($65.66) into a new `fare_capped` column to minimize skewness while preserving distribution shapes.
  - **Engineered Features**:
    - `FamilySize`: Combined family count (`sibsp` + `parch` + `1`).
    - `IsAlone`: Binary categorization (1 if passenger is alone, 0 otherwise).
    - `FarePerPerson`: Individual ticket cost (`fare / FamilySize`).

###  Task 3: Statistical Analysis & Business Insights
- **File**: [`project1/task3.py`](file:///c:/Users/KIIT/OneDrive/Desktop/DecodeLabs-Internship/project1/task3.py)
- **Objective**: Extract key statistics and demographical patterns related to passenger survival.
- **Key Features**:
  - **General Metrics**: Computes overall survival rate (38.38%), passenger counts by gender and class, average age, and median fare.
  - **Correlation Matrix**: Generates a Pearson correlation matrix showing variables most associated with survival (e.g. female gender: `+0.54`, higher class: `-0.33`, traveling alone: `-0.20`).
  - **Age Binning**: Categories passengers into five age groups (Child, Teenager, Young Adult, Adult, Senior) to compute survival rates by age bracket.
  - **Statistical Insights**: Detailed explanations of 5 distinct insights:
    1. *Gender-based Survival*: Strong enforcement of "women and children first" (74.2% female survival vs. 18.9% male).
    2. *Class-based Survival*: Significant advantage of first class (62.6%) vs. third class (24.2%).
    3. *Class & Gender Interaction*: 96.8% of first-class females survived compared to only 13.5% of third-class males.
    4. *Alone vs. Accompanied*: Traveling with family increased survival likelihood to 50.6% (compared to 30.1% for solo travelers).
    5. *Embarkation Port Influence*: Passengers from Cherbourg (C) had a higher survival rate (55.4%) due to a higher proportion of first-class tickets.

###  Task 4: Comprehensive Data Visualization
- **File**: [`project1/task4.py`](file:///c:/Users/KIIT/OneDrive/Desktop/DecodeLabs-Internship/project1/task4.py)
- **Objective**: Generate and save high-resolution plots illustrating dataset characteristics and findings.
- **Key Features**:
  - **Unified Design System**: Sets custom styles, palettes, labels, and fonts using `seaborn` and `matplotlib`.
  - **Automated Plot Generation**: Saves the following 9 visualizations in the `project1/visualizations/` directory:
    - `plot1_survival_gender.png`: Survival rate comparison between male and female passengers.
    - `plot2_survival_class.png`: Survival rate comparison across passenger classes.
    - `plot3_survival_agegroup.png`: Survival rate by age category.
    - `plot4_age_distribution.png`: Histogram displaying the distribution of passenger ages after imputation.
    - `plot5_fare_distribution.png`: Comparative distributions showing original fares vs. IQR capped fares.
    - `plot6_correlation_heatmap.png`: Heatmap illustrating correlation coefficients between numerical variables.
    - `plot7_boxplots.png`: Age and capped fare distributions stratified by passenger class.
    - `plot8_categorical_counts.png`: Grid plots of passenger counts across gender, class, embarkation port, and travel status.
    - `plot9_scatter_age_fare.png`: Scatter plot mapping age vs. fare, color-coded by survival outcome.

---

## How to Run the Tasks

Then, run any task from the repository root:

```bash
# Run Task 1
python project1/task1.py

# Run Task 2
python project1/task2.py

# Run Task 3
python project1/task3.py

# Run Task 4 (generates and saves plots)
python project1/task4.py
```
