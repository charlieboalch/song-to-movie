import csv
import json
import os

import pandas as pd
from transformers import pipeline

from scrape_movies import fetch_movie
from util.models import run_embeddings, get_valence
from util.vectors import MediaVector

emotion_model = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

emotion_map = {
    'admiration': [0, 0, -0.1, 0, 0.2, 0],
    'amusement': [0.1, 0.05, 0, 0, 0, 0.1],
    'anger': [-0.1, 0, 0.1, 0, -0.1, 0],
    'optimism': [0.2, 0, -0.1, 0, 0, 0],
    'sadness': [-0.1, 0, 0, 0, 0, 0],
    'love': [0, 0, 0, 0, 0.1, 0],
    'surprise': [0.05, 0, 0, 0.1, 0, 0],
    'joy': [0.1, 0.05, 0, 0, 0, 0.02],
    'excitement': [0, 0.1, 0, 0.05, 0, 0.02],
    'desire': [0, 0, 0, 0.05, 0.1, 0],
    'disgust': [-0.05, 0, 0.05, 0, -0.1, 0],
    'fear': [0, 0.05, 0.05, 0.1, 0, 0],
    'nervousness': [0, 0, 0, 0.1, 0, 0],
    'annoyance': [0, 0, 0.05, 0.05, -0.05, 0],
    'embarrassment': [0, 0, 0, 0.1, 0, 0],
    'disappointment': [-0.05, 0, 0, 0, 0, 0],
    'curiosity': [0, 0.05, 0, 0.05, 0, 0],
    'confusion': [0, 0, 0, 0.05, 0, 0],
    'remorse': [0, 0, -0.1, 0, 0.05, 0],
    'caring': [0.05, 0, -0.05, 0, 0.1, 0]
}

genre_map = {
    'Romance': [0, 0, 0, 0.05, 0.15, 0],
    'Crime': [0, 0.1, 0.1, 0, 0, 0],
    'Drama': [0, -0.1, 0, 0.1, 0, 0],
    'Action': [0, 0.2, 0, 0.05, 0, 0],
    'Comedy': [0.2, 0, 0, 0, 0, 0.2],
    'Adventure': [0, 0.15, 0, 0, 0, 0],
    'Thriller': [0, 0.1, 0.05, 0.2, 0, 0],
    'Mystery': [0, 0, 0, 0.2, 0, 0],
    'War': [0, 0.05, 0.15, 0, -0.1, 0],
    'Horror': [0, 0, 0.25, 0.05, -0.1, 0],
    'Family': [0.1, 0, 0, 0, 0.15, 0]
}

def chunk_plot(plot: str) -> list[str]:
    chunked_plot = []
    split = plot.split(".")

    if len(split) <= 2:
        return split

    for i in range(len(split) - 2):
        chunked_plot.append('. '.join(split[i:i+2]))

    return chunked_plot

def keyword_emotions(keywords, movie_vector: MediaVector):
    for i in keywords:
        result = emotion_model(i)[0]
        if not result[0]['label'] == 'neutral' and result[0]['label'] in emotion_map:
            # adjust the movie vector based on detected keyword
            movie_vector.adjust_rankings(emotion_map[result[0]['label']])

    return movie_vector

# adjust movie vector based on preset genre scalars
def genre_tuning(genres, movie_vector: MediaVector):
    for i in genres:
        if i in genre_map:
            movie_vector.adjust_rankings(genre_map[i])

    return movie_vector

# load movie by tmdb id and generate a movie vector
def analyze_movie(movie, id):
    print(f"{movie['title']} {id}")

    # 1. chunk plot into groups of sentences
    chunked = chunk_plot(movie['plot'])

    # 2. get bulk of analysis from plot
    embeddings_data = run_embeddings(chunked)
    valence = get_valence(chunked)

    vector = MediaVector(valence, embeddings_data[0], embeddings_data[1],
                         embeddings_data[2], embeddings_data[3], embeddings_data[4])

    # 3. fine tuning based on user defined keywords
    vector = keyword_emotions(movie['keywords'], vector)

    # 4. additional tuning based on genres
    vector = genre_tuning(movie['genres'], vector)

    # save as a file
    with open(f"cache/vectors/{id}.movie", "w") as f:
        f.writelines(json.dumps(vector, default=vars))

    return vector.to_list()

if __name__ == '__main__':
    # parse all cached movies in the movie directory
    movie_list = [x.replace(".json", "") for x in os.listdir("../cache/movies")]

    # csv file columns
    columns = [["movie", "valence", "energy", "darkness", "tension", "warmth", "humor"]]

    # load and analyze
    for i in movie_list:
        movie_info = fetch_movie(i)
        if movie_info is None:
            continue

        scored_data = analyze_movie(movie_info, i)
        movie_title = f'"{movie_info['title']}"'
        columns.append([movie_title, scored_data[0], scored_data[1], scored_data[2],
                        scored_data[3], scored_data[4], scored_data[5]])

    # save to csv
    with open('../data/movies.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(columns)

    # reload as dataframe to z score
    df = pd.read_csv("../data/movies.csv", encoding='utf-8')

    cols = ['valence', 'energy', 'darkness', 'tension', 'warmth', 'humor']
    movie_stats = {}

    for col in cols:
        mean = df[col].mean()
        std = df[col].std()
        movie_stats[col] = (mean, std)
        df[col] = (df[col] - mean) / std

    df.to_csv('data/movies-z.csv')
    with open('../data/movies-stats.json', 'w') as f:
        f.writelines(json.dumps(movie_stats))