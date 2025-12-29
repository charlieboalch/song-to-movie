import json

from fastapi import FastAPI, HTTPException, Request
from starlette.responses import StreamingResponse

from ranker import MovieRanker
from score_song import score_song

app = FastAPI()
ranker = MovieRanker()


@app.get('/hello')
def hello():
    return 'hello, world'


async def generate_vectors(request, song_vectors: list[str]):
    average_vector = [0] * 6

    for song_id in song_vectors:
        if await request.is_disconnected():
            print("Client disconnected, stopping stream")
            await request.close()
            return

        title, vector = score_song(song_id)

        for i in range(len(vector)):
            average_vector[i] += (vector[i] * (1 / len(song_vectors)))

        yield f"data: {json.dumps({'track': title, 'vector': vector})}\n\n"

    movie_titles, movie_scores, movie_urls = ranker.top_k_movies(average_vector, k=6)
    movie_results = []
    for i in range(len(movie_titles)):
        movie_results.append({'movie': movie_titles[i], 'score': movie_scores[i], 'url': movie_urls[i]})

    yield f"data: {json.dumps({'movies': movie_results})}\n\n"
    await request.close()



@app.get('/rank_movies')
async def rank_movies(request: Request, songs: str = ''):
    songs = songs.split(',')
    if len(songs) == 0:
        raise HTTPException(status_code=400, detail='Invalid request format')

    return StreamingResponse(generate_vectors(request, songs), media_type="text/event-stream")
