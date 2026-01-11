import pandas as pd
def load_data(filepath):
    """
    This function Reads the raw CSV file and returns a pandas DataFrame.
    This keeps data loading logic in one place.
    """
    return pd.read_csv(filepath)

# Basic data cleaning
def clean_data(df):
    """
    Cleans the HR dataset by: Removing irrelevant columns, Handling missing values, Encoding the target variable (Attrition)
    """
    # After reviewing data we found 3 columns that have only 1 unique value and 1 column with 1470 unique values (EmployeeNumber) so we remove them
    df = df.drop(columns=[
        "EmployeeNumber",
        "Over18",
        "EmployeeCount",
        "StandardHours"
    ])

    # After inspection, we know that the data is clean but for future we can still define these parts
    # Filling missing numerical values using median
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Handle categorical missing values safely
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Encode target variable
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

    return df

# Identifying  categorical columns (LOW cardinality)

def get_categorical_columns(df, max_unique=30):
    """
    Identifies categorical (object-type) columns
    with limited unique values. We will use it for EDA and encoding decisions.
    """

    object_cols = []

    for column in df.columns:
        if df[column].dtype == object and df[column].nunique() <= max_unique:
            object_cols.append(column)

    return object_cols


# Identifying discrete numerical columns
def get_discrete_columns(df, max_unique=30):
   #Identifies numerical columns that behave like categories.
    discrete_cols = []

    for column in df.columns:
        if df[column].dtype != object and df[column].nunique() < max_unique:
            discrete_cols.append(column)

    return discrete_cols


# Encoding categorical features

def encode_features(df, categorical_cols):
    """
    Applies one-hot encoding to categorical columns.
    drop_first=True helps avoid multicollinearity.
    """
    return pd.get_dummies(df, columns=categorical_cols, drop_first=True)



# Feature engineering
def feature_engineering(df):

    #Creates new features that may improve model performance.

    # Monthly income adjusted by job level
    df["Income_per_JobLevel"] = df["MonthlyIncome"] / df["JobLevel"]

    # Company experience relative to employee age
    df["YearsAtCompany_per_Age"] = df["YearsAtCompany"] / df["Age"]

    return df
