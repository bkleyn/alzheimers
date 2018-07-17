import numpy as np
import pandas as pd
import re
import time

# ==================================
# Import base data
# ==================================

print("Importing base table...")

# Read data
data = pd.read_csv('../data/tadpole/TADPOLE_D1_D2.csv', low_memory=False)

# Set index
data.set_index(['RID', 'VISCODE'], inplace=True, verify_integrity=True)

# ==================================
# ADAS score data (ADASSCORES.csv)
# ==================================

print("Importing ADAS scores...")

# Read data
data_adas = pd.read_csv('../data/study_data/ADASSCORES.csv')

# Set index
data_adas.set_index(['RID', 'VISCODE'], inplace=True, verify_integrity=True)

# Drop unnecessary columns
cols = set(data_adas.columns) - set(data_adas.columns[5:20])
data_adas.drop(cols, axis=1, inplace=True)

# Rename columns
data_adas.columns = ["ADAS_" + c for c in data_adas.columns]

# Join onto base data
data = pd.merge(data, data_adas, how="left", left_index=True, right_index=True)

# ==================================
# Patient demographics (PTDEMOG.csv)
# ==================================

print("Importing patient demographics...")

# NOTES: Should create a "age retired" feature once joined onto base table

# Read data
data_demo = pd.read_csv('../data/study_data/PTDEMOG.csv')

# Get latest record by patient
g = data_demo.sort_values(['RID', 'USERDATE']).groupby(['RID'])
data_demo['RNK'] = g['USERDATE'].rank(ascending=False)

# List of columns to keep
cols = ['RID',
        'PTHAND',           # Handedness (left, right)
        'PTWORK',           # Primary occupation as adult
        'PTWRECNT',         # Most recent occupation
        'PTNOTRT',          # Retired
        'PTRTYR',           # Retirement date
        'PTHOME',           # Type of residence
        'PTTLANG',          # Language used for tests
        'PTPLANG']          # Primary language

# Only keep columns of interest
data_demo = data_demo.loc[data_demo['RNK'] == 1, cols]

#Set index
data_demo.set_index('RID', inplace=True, verify_integrity=True)

#Join onto base data
data = data.join(data_demo, how="left")

# ==================================
# Family History (FHQ.csv)
# ==================================

print("Importing family history data...")

# Read data
data_fam = pd.read_csv('../data/study_data/FHQ.csv')

# Get 'screening' record for each patient
data_fam = data_fam.loc[data_fam['VISCODE'] == "sc", :]

# List of columns to keep
cols = ['RID',
        'FHQPROV',          # Family history provider
        'FHQMOM',           # Mother Dementia
        'FHQMOMAD',         # Mother Alzheimer's
        'FHQDAD',           # Father Dementia
        'FHQDADAD',         # Father Alzheimer's
        'FHQSIB']           # Does the patient have siblings

# Only keep columns of interest
data_fam = data_fam.loc[:, cols]

#Set index
data_fam.set_index('RID', inplace=True, verify_integrity=True)

#Join onto base data
data = data.join(data_fam, how="left")

# ==================================
# TOMM40 polymorphism (TOMM40.csv)
# ==================================

print("Importing TOMM40 data...")

# Read data
data_tom = pd.read_csv('../data/study_data/TOMM40.csv')

#Set index
data_tom.set_index('RID', inplace=True, verify_integrity=True)

# Only keep columns of interest
data_tom = data_tom.loc[:, ['TOMM40_A1', 'TOMM40_A2']]

#Join onto base data
data = data.join(data_tom, how="left")

# ==================================
# Import metadata file
# ==================================

print("Importing metadata...")

data_meta = pd.read_csv('../data/metadata_raw.csv')

# Drop entries for columns that are not going to be used
data_meta = data_meta.loc[data_meta['keep'] == 1, :]

# Only keep columns flagged in metadata
cols = list(data_meta.column_name)
data = data[cols]

# ==================================
# Data cleaning (numerical features)
# ==================================

print("Cleaning numerical features...")

# Replace missing/blanks with NA
data.replace(' ', np.nan, inplace=True)

# Numeric factors
cols = ['ABETA_UPENNBIOMK9_04_19_17',
        'TAU_UPENNBIOMK9_04_19_17', 
        'PTAU_UPENNBIOMK9_04_19_17']

for c in cols:
    data[c] = data[c].str.replace("<", "").str.replace(">", "")

numeric_cols = list(data_meta.loc[(data_meta['numeric'] == 1)
                                  & (data_meta['data_type'] == 'object'), 
                                  'column_name'])
for c in numeric_cols:
    data[c] = data[c].apply(pd.to_numeric)

# ==================================
# Export data for visualization
# ==================================

data_for_viz = data.copy(deep=True)


data_for_viz['PTRTYR'] = data_for_viz['PTRTYR'].apply(lambda x: str(x)[-4:])
data_for_viz['PTRTYR'] = data_for_viz['PTRTYR'].apply(
    pd.to_numeric, errors='coerce')

data_for_viz.to_csv('../data/data_all_no_encode.csv')

# ==================================
# Data cleaning (categorical features)
# ==================================

print("Cleaning categorical features...")

def clean_string(row):
    try:
        s = row.lower().replace(" ", "_").replace("/", "_").replace(".", "")
        s = s.replace("-4", "unknown")
    except:
        s = None

    return s

# Non-numeric categorical factors
cat_cols = list(data_meta.loc[(data_meta['categorical'] == 1)
                              & (data_meta['data_type'] == 'object'), 
                              'column_name'])

for c in cat_cols:

    # Clean text
    data[c] = data[c].apply(clean_string)

    # Group smaller levels
    counts = list(pd.value_counts(data[c]).index[:30])
    data[c] = data[c].apply(lambda x: x if x in counts else 'other')

    # Create dummy variables
    df = pd.get_dummies(data[c], prefix=c, drop_first=True)
    data = data.join(df)

    # Drop original column
    data.drop(c, axis=1, inplace=True)

# Numeric categorical factors
cat_cols = list(data_meta.loc[(data_meta['categorical'] == 1)
                              & (data_meta['data_type'] != 'object'), 
                              'column_name'])

for c in cat_cols:

    # Change type
    data[c] = data[c].apply(pd.to_numeric, downcast='integer')

    # Create dummy variables
    df = pd.get_dummies(data[c], prefix=c, drop_first=True)
    data = data.join(df)

    # Drop original column
    data.drop(c, axis=1, inplace=True)


# ==================================
# Time factors
# ==================================

# Retirement age
data['PTRTYR'] = data['PTRTYR'].apply(lambda x: str(x)[-4:])
data['PTRTYR'] = data['PTRTYR'].apply(pd.to_numeric, errors='coerce')

# Output cleaned data
data.to_csv('../data/data_all.csv')

print("Done.")