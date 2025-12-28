from flask import Flask, request, abort

from ranker import MovieRanker
from score_song import score_song

app = Flask(__name__)
ranker = MovieRanker()

@app.route('/hello')
def hello():
    return 'hello, world'

@app.route('/rank_movies', methods=['GET'])
def rank_movies():
    songs = request.args.get('songs', '').split(',')
    if len(songs) == 0:
        abort(400)

    results = {'songs': [], 'movies': []}
    average_vector = [0] * 6

    for song_id in songs:
        title, vector = score_song(song_id)
        results['songs'].append({'track': title, 'vector': vector})

        for i in range(len(vector)):
            average_vector[i] += (vector[i] * (1 / len(songs)))

    movie_titles, movie_scores = ranker.top_k_movies(average_vector)
    for i in range(len(movie_titles)):
        results['movies'].append({'movie': movie_titles[i], 'score': movie_scores[i]})

    return results
