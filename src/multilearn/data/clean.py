from mastml.feature_generators import ElementalFeatureGenerator

import pandas as pd
import glob
import os

feature_types = [
                 'composition_avg',
                 'arithmetic_avg',
                 'max',
                 'min',
                 'difference',
                 ]

path = './original'
out = './outputs'

files = glob.glob(os.path.join(path, '*.csv'))
os.makedirs(out, exist_ok=True)

for i in files:

    df = pd.read_csv(i)
    df, _ = ElementalFeatureGenerator(
                                      df['comp'],
                                      feature_types=feature_types,
                                      remove_constant_columns=True,
                                      ).evaluate(X=df, y=df['y'])
    df = df.drop(['comp'], axis=1)

    df.to_csv(os.path.join(out, i.split('/')[-1]), index=False)
    print(df)
