from ranker import MovieRanker
from score_song import score_song

spotify_id = ['TCACP1639765']
ranker = MovieRanker()

for sid in spotify_id:
    song, song_vec = score_song(sid)
    movies, scores = ranker.top_k_movies(song_vec)

    print(f'Movies similar to {song}:')
    for i in range(len(movies)):
        print(f'{movies[i]} ({scores[i]})')
    print()

