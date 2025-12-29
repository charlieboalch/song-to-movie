import csv
import json
import os

import pandas as pd

df = pd.read_csv("data/movies-url.csv", encoding='utf-8')

cols = ['valence', 'energy', 'darkness', 'tension', 'warmth', 'humor']
movie_stats = {}

for col in cols:
    mean = df[col].mean()
    std = df[col].std()
    movie_stats[col] = (mean, std)
    df[col] = (df[col] - mean) / std

df.to_csv('movies-z.csv')
with open('data/movies-stats.json', 'w') as f:
    f.writelines(json.dumps(movie_stats))