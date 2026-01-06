import csv
import json
import os

import dotenv
import pandas as pd
import requests

def generate_stats():
    df = pd.read_csv("data/movies-url.csv", encoding='utf-8')

    cols = ['valence', 'energy', 'darkness', 'tension', 'romance', 'humor']
    movie_stats = {}

    for col in cols:
        mean = df[col].mean()
        std = df[col].std()
        movie_stats[col] = (mean, std)
        df[col] = (df[col] - mean) / std

    df.to_csv('movies-z.csv')
    with open('data/movies-stats.json', 'w') as f:
        f.writelines(json.dumps(movie_stats))

def load_urls():
    dotenv.load_dotenv()
    TMDB_API = os.getenv('TMDB_API')
    OMDB_API = os.getenv('OMDB_API')

    TMDB_URL = "https://api.themoviedb.org/3/movie/"
    OMDB_URL = f"http://www.omdbapi.com/?plot=full&apikey={OMDB_API}&t="

    auth_header = {'Authorization': f"Bearer {TMDB_API}"}

    columns = [["movie", "valence", "energy", "darkness", "tension", "romance", "humor", "url"]]
    cols = ['valence', 'energy', 'darkness', 'tension', 'romance', 'humor']

    for file in os.listdir('cache/vectors'):
        print(file)
        output = [''] * 8

        with open(f'cache/vectors/{file}', 'r') as f:
            vector = json.loads(''.join(f.readlines()))

        for i in range(len(cols)):
            output[i + 1] = vector[cols[i]]

        with open(f"cache/movies/{file.replace('.movie', '.json')}", "r") as f:
            data = json.loads(''.join(f.readlines()))

        output[0] = data['title']

        r = requests.get(TMDB_URL + file.replace(".movie", ""), headers=auth_header)
        r = r.json()
        output[7] = r['poster_path']
        columns.append(output)

    with open('data/movies-url.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(columns)
