from sklearn.model_selection import train_test_split
import pandas as pd
import glob
import os

path = './outputs'
out = os.path.join(path, 'splits')

files = glob.glob(os.path.join(path, '*.csv'))
os.makedirs(out, exist_ok=True)

for i in files:

    df = pd.read_csv(i)

    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
    df_val, df_test = train_test_split(df_val, test_size=0.5, random_state=42)

    df_train.to_csv(os.path.join(out, 'train_'+i.split('/')[-1]), index=False)
    df_val.to_csv(os.path.join(out, 'val_'+i.split('/')[-1]), index=False)
    df_test.to_csv(os.path.join(out, 'test_'+i.split('/')[-1]), index=False)
