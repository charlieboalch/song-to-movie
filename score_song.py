import json
import os
import re

import dotenv
import requests
from lyricsgenius import Genius

from util.models import run_embeddings, get_valence

dotenv.load_dotenv()
genius = Genius(os.getenv('GENIUS_API'))
genius.verbose = False

with open('data/movies-stats.json', 'r') as f:
    movie_stats = json.loads(''.join(f.readlines()))

cols = ['valence', 'energy', 'darkness', 'tension', 'warmth', 'humor']

# fetch track metadata given a spotify ID / IRSC ID
def fetch_recco_id(spotify_id):
    base_url = " https://api.reccobeats.com/v1/track?ids=" + spotify_id
    response = requests.get(base_url).json()

    recco_data = response['content'][0]
    return recco_data['id'], recco_data['trackTitle'], recco_data['artists'][0]['name']

# fetch track audio features
def fetch_song_features(recco_id):
    url = f"https://api.reccobeats.com/v1/track/{recco_id}/audio-features"

    r = requests.get(url)
    return r.json()

# analyze audio features
def analyze_features(song_features):
    valence = song_features['valence']

    # take tempo into account too
    energy = song_features['energy'] * 0.7 + (0.3 * (song_features['tempo'] / 110))

    # darkness - take loudness, mood, and mode into account (low loudness, low mood, minor chord)
    darkness = ((abs(song_features['loudness']) / 60 * 0.5) * ((1 - valence) * 0.3)
                + (0.25 if song_features['mode'] == 0 else 0))

    # tension - take tempo, mode, and danceability (high tempo, minor chord, high danceability)
    tension = ((0.5 * (song_features['tempo'] / 110)) + (0.2 if song_features['mode'] == 0 else 0)
               + (0.3 * song_features['danceability']))

    # warmth - take acousticness, mode, and tempo into account (high acousticness, major chord, slower tempo)
    warmth = ((0.5 * song_features['acousticness']) + (0.2 if song_features['mode'] == 1 else 0)
              + (0.1 * (110 / song_features['tempo'])))

    # humor - take valence, energy, loudness into account (high everything)
    humor = (0.7 * valence) + (0.2 * energy) + (0.1 * (-1 * abs(song_features['loudness']) / 60 + 1))

    return [valence, energy, darkness, tension, warmth, humor]

# analyze lyrics using roberta and sbert
def analyze_lyrics(track_title, track_artist):
    # fetch lyrics
    song = genius.search_song(track_title, track_artist)

    # return if song not found or no lyrics
    if song is None:
        return None

    lyrics = re.sub(r'\[.*]', '', song.lyrics).split('\n')
    lyrics = [x for x in lyrics if len(x) != 0]

    # split lyrics into chunks of 3 overlapping sentences
    chunked_lyrics = []
    if len(lyrics) <= 3:
        chunked_lyrics = lyrics
    else:
        for i in range(len(lyrics) - 3):
            if not i % 2 == 0:
                continue

            chunked_lyrics.append('. '.join(lyrics[i:i+3]))

    # get the embedding info and the valence
    embedding = run_embeddings(chunked_lyrics)
    valence = get_valence(chunked_lyrics)

    output = [valence]
    output.extend(embedding)

    return output

# generate the vector of a song
def generate_song_vector(spotify_id):
    # check cache
    if os.path.exists(f"cache/songs/{spotify_id}.json"):
        with open(f"cache/songs/{spotify_id}.json") as f:
            data = json.loads(''.join(f.readlines()))
            return data['title'], data['vector']

    # fetch song data
    recco_id, track_title, artist = fetch_recco_id(spotify_id)
    song_features = fetch_song_features(recco_id)

    # generate vectors from two different sources
    feature_vector = analyze_features(song_features)
    lyric_vector = analyze_lyrics(track_title, artist)

    # combine the two with 40% lyrics and 60% audio features
    combined_vector = [0] * 6
    for i in range(len(feature_vector)):
        if lyric_vector is not None:
            combined_vector[i] = lyric_vector[i - 1] * 0.5 + (feature_vector[i] - 0.4 + movie_stats[cols[i]][0]) * 0.5
        else:
            combined_vector[i] = feature_vector[i] - 0.4 + movie_stats[cols[i]][0]

    # write to cache
    with open(f"cache/songs/{spotify_id}.json", "w") as f:
        output_data = {
            'title': track_title,
            'vector': combined_vector
        }
        f.writelines(json.dumps(output_data))

    return track_title, combined_vector

# adjust vector to the z score
def score_song(spotify_track):
    title, vector = generate_song_vector(spotify_track)

    # find z score of vector columns
    for i in range(len(cols)):
        mean, std = movie_stats[cols[i]]
        vector[i] = (vector[i] - mean) / std

    return title, vector