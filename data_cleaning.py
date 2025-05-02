import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


df = pd.read_csv('real_estate_thesis_pois.csv', low_memory=False)


df.drop(columns=['Unnamed: 0'], inplace=True)


df['date_created'] = pd.to_datetime(df['date_created'])
df['date_created_first'] = pd.to_datetime(df['date_created_first'])
df['pushed_up_at'] = pd.to_datetime(df['pushed_up_at'], errors='coerce', utc=True)


df['price_per_m2'] = df['price']/df['area_m2']


df.drop_duplicates(subset=['id'], inplace=True)

df.set_index('id', inplace=True)


# we only have SELL type data
df = df[df['transaction_type']=="SELL"] #.reset_index(drop=True)

df.drop(columns=['transaction_type', 'date_created', 'date_created_first', 'pushed_up_at', 'count', 'geometry'], inplace=True)

df.rename(columns={'location.district': 'district', 'location.latitude': 'latitude',
       'location.longitude': 'longitude', 'location.neighborhood': 'neighborhood'}, inplace=True)


# remove outliers, where price=1, etc
df = df[df['estate']=="FLAT"] #.reset_index(drop=True)
df.drop(columns=['estate'], inplace=True)

# filter out
cond1 = df['price_per_m2']>2000
cond2 = df['area_m2']>=10

# remove outliers
#cond3 = df['area_m2'] < df['area_m2'].mean() + df['area_m2'].std()*3
#cond4 = df['price'] < df['price'].mean() + df['price'].std()*3
#cond5 = df['building_floors_num']< df['building_floors_num'].mean() + df['building_floors_num'].std()*3
#cond6 = df['price_per_m2']< df['price_per_m2'].mean() + df['price_per_m2'].std()*3

df = df[cond1&cond2] #.reset_index(drop=True)


panorama = pd.read_csv('panorama_districts.csv', index_col=0)
panorama.reset_index(inplace=True)
panorama['pano_share_pop_females'] = panorama['pano_pop_females'] / panorama['pano_pop_total']


#df = pd.merge(df, panorama, on='district', how='left')
df = df.reset_index()
df = df.merge(panorama, on='district', how='left')
df.set_index('id', inplace=True)



# Data transformation


floor_mapping = {
    'CELLAR': -1,
    'GROUND': 0,
    'FIRST': 1,
    'SECOND': 2,
    'THIRD': 3,
    'FOURTH': 4,
    'FIFTH': 5,
    'SIXTH': 6,
    'SEVENTH': 7,
    'EIGHTH': 8,
    'NINTH': 9,
    'TENTH': 10,
    'ABOVE_TENTH': 11,
    'GARRET': 12
}


df['floor_number'] = df['floor_number'].map(floor_mapping)

df['building_year'] = np.select([df['building_year']<1800, df['building_year']>2028], [np.nan, np.nan], default=df['building_year'])

df['building_age'] = pd.Timestamp.now().year - df['building_year']

city_center = (52.2318543, 21.0028622)  # for Warsaw

# 1 degree of latitude/longitude ≈ 111 km
df['distance_to_center_km'] = df.apply(
    lambda row: np.sqrt(
        (row['latitude'] - city_center[0])**2 +
        (row['longitude'] - city_center[1])**2
    ) * 111,
    axis=1
)



df['room_size_m2'] = df['area_m2'] /  df['rooms_number'].replace('MORE', 11).astype(float)
df['floor_ratio'] = df['floor_number'] / df['building_floors_num']





top_heating = df["heating"].value_counts().index[:4]
df["heating"] = df["heating"].apply(lambda x: x if x in top_heating else "OTHER")



top_building_type = df["building_type"].value_counts().index[:3]
df["building_type"] = df["building_type"].apply(lambda x: x if x in top_building_type else "OTHER")


df["building_material"] = df["building_material"].replace("[]", "MISSING")
top_building_material = df["building_material"].value_counts().index[:6]
df["building_material"] = df["building_material"].apply(lambda x: x if x in top_building_material else "OTHER")




df['rooms_number_original'] = df['rooms_number'].copy()
df['rooms_number_original'] = df['rooms_number_original'].astype('str').replace('MORE', '11').astype('Int64')

df['rooms_number'] = df['rooms_number'].astype('str').replace('5|6|7|8|9|10', 'MORE', regex=True)
category_order = ['1', '2', '3', '4', 'MORE']
df['rooms_number'] = pd.Categorical(df['rooms_number'], categories=category_order, ordered=True)





### remove outliers


import numpy as np

multiplier = 1.5
columns = ['area_m2', 'price', 'building_floors_num', 'price_per_m2', 'room_size_m2']

df["outlier"] = False  # Initialize all as non-outliers

for col in columns:
    if col in ['price', 'price_per_m2']:
        values = np.log(df[col])
    else:
        values = df[col]

    Q1 = values.quantile(0.25)
    Q3 = values.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    if col in ['price', 'price_per_m2']:
        mask = (np.log(df[col]) < lower_bound) | (np.log(df[col]) > upper_bound)
    else:
        mask = (df[col] < lower_bound) | (df[col] > upper_bound)

    df.loc[mask, "outlier"] = True  # Mark outliers

print(df["outlier"].sum(), "rows marked as outliers.")




# remove columns with only one unique val and those that are redundant
cols_to_drop = ['park_areas_2000_m2', 'park_areas_1000_m2']
for col in df.columns:
    if len(df[col].unique())==1:
        cols_to_drop.append(col)

df.drop(columns=cols_to_drop, inplace=True)
print("Removed columns", str(cols_to_drop), "with one unique value")



from scipy.stats import skew, shapiro


numerical_columns = [col for col in df.select_dtypes(include=[np.number]).columns if 'has_' not in col and 'vector_' not in col]






skewness = df[numerical_columns].apply(lambda x: skew(x.dropna()))
print("Skewness of numerical variables:")
print(skewness)

for col in numerical_columns:
    stat, p = shapiro(df[col].dropna())
    print(f'Shapiro-Wilk test for {col}: statistic={stat}, p-value={p}')





### Data normalization

log_transform_cols = ['area_m2', 'floor_number', 'building_age',
                       'building_floors_num', 'distance_to_center_km', 'room_size_m2', 'floor_ratio' ]

scale_cols = ['latitude', 'longitude',  ]

if 'building_year' in df.columns:
    df.drop(columns=['building_year'], inplace=True)




numeric_cols = df.select_dtypes(include=np.number).columns


df['has_street_name'] = df['location.name'].notna()
df['has_street_number'] = df['location.number'].notna()



numeric_cols = df.select_dtypes(include=["number"]).columns




# imputation


def fill_na_excluding_outliers(x):
    mask = ~x.index.map(df['outlier'])
    mean_val = x[mask].mean()
    return x.fillna(mean_val)

df[numeric_cols] = df.groupby("district")[numeric_cols].transform(fill_na_excluding_outliers)




# for panorama values it makes sense to fill with 0
df[['pano_parks_n', 'pano_parks_area', 'pano_parks_avg_area',
    'pano_bushes_loss', 'pano_cemeteries_n', 'pano_cemeteries_area']] = df[['pano_parks_n', 'pano_parks_area', 'pano_parks_avg_area',
                                                                           'pano_bushes_loss', 'pano_cemeteries_n', 'pano_cemeteries_area']].fillna(0)



numeric_cols = numeric_cols.drop(['price', 'price_per_m2', ]) # 'price_bins'

# scaling


from sklearn.preprocessing import RobustScaler
import joblib
import numpy as np


non_outlier_data = df.loc[~df['outlier'], numeric_cols]

# Dictionary to store scalers
scalers = {}

# Fit a scaler for each column
for col in ['area_m2', 'building_age', 'rooms_number_original', 'room_size_m2', 'pano_transport_buses', 'pano_share_pop_females', 'distance_to_center_km']:
    scaler = RobustScaler()
    # Fit on a single column, reshaped to 2D array as required by sklearn
    scaler.fit(non_outlier_data[[col]])  # Double brackets keep it as a DataFrame
    scalers[col] = scaler

# Save all scalers to a single file
joblib.dump(scalers, 'scalers_dict.pkl')





from sklearn.preprocessing import RobustScaler


from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np

import joblib

scaler = RobustScaler()
non_outlier_data = df.loc[~df['outlier'], numeric_cols]
scaler.fit(non_outlier_data)

joblib.dump(scaler, 'robust_scaler.pkl')

df[numeric_cols] = scaler.transform(df[numeric_cols])


print("Scaled data statistics:")
df[numeric_cols].describe()



# one hot encoding


df.drop(columns=['location.name', 'location.number', 'neighborhood', 'has_street_name', 'has_street_number', ], inplace=True)

if 'price_bins' in df.columns and 'price_range' in df.columns:
    df.drop(columns=['price_bins','price_range'], inplace=True)






categorical_columns = df.select_dtypes(include=['object', 'category']).columns

print("We have " + str(len(categorical_columns)) + " columns of type object to encode")
print("Names of the columns that needs to be encoded: "+ "\n" + str(categorical_columns))
df.select_dtypes(include=['object', 'category']).head()





import unicodedata
import re
from unidecode import unidecode


def clean_text(text):
    """
    Clean and normalize text to handle special characters and symbols
    """
    if pd.isna(text):
        return text

    text = str(text).lower()

    text = unidecode(text)

    text = re.sub(r'[^a-z0-9]+', '_', text)

    text = re.sub(r'_+', '_', text)
    text = text.strip('_')

    return text



for i in range(0,len(categorical_columns)):
    column = categorical_columns[i]
    df[column] = df[column].apply(clean_text)
    df = pd.concat([df,pd.get_dummies(df[column],prefix=column)],axis=1).drop([column],axis=1)

df.shape




df = df[df['price'].notna()] #.reset_index(drop=True)



# drop properties outside of warsaw
indices_to_drop = [65349901, 65949392, 66357797, 65818114, ]

df = df.drop(indices_to_drop, axis=0)


df.to_csv('real_estate_thesis_processed.csv')