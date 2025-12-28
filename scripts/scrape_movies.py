import json
import os

import dotenv
import requests

dotenv.load_dotenv()
TMDB_API = os.getenv('TMDB_API')
OMDB_API = os.getenv('OMDB_API')

TMDB_URL = "https://api.themoviedb.org/3/movie/"
OMDB_URL = f"http://www.omdbapi.com/?plot=full&apikey={OMDB_API}&t="

auth_header = {'Authorization': f"Bearer {TMDB_API}"}

def fetch_movie(id: str):
    if os.path.exists(f"cache/movies/{id}.json"):
        with open(f"cache/movies/{id}.json") as f:
            data = json.loads(''.join(f.readlines()))
            return data

    r = requests.get(TMDB_URL + id, headers=auth_header)
    gen_data = r.json()

    r = requests.get(TMDB_URL + id + "/keywords", headers=auth_header)
    keyword_data = r.json()

    normal_title = gen_data['title'].lower().replace(" ", "+")

    r = requests.get(OMDB_URL + normal_title)
    plot_data = r.json()

    if 'Plot' not in plot_data:
        print(f"failed to fetch {normal_title}")
        return None

    final_data = {
        "title": gen_data['title'],
        "genres": list(map(lambda x: x['name'], gen_data['genres'])),
        "plot": plot_data['Plot'],
        "summary": gen_data['overview'],
        "keywords": list(map(lambda x: x['name'], keyword_data['keywords'])),
        "runtime": gen_data['runtime']
    }

    with open(f"cache/movies/{id}.json", "w") as f:
        f.writelines(json.dumps(final_data))

    return final_data

def scrape_movie_page(page):
    url = f'https://api.themoviedb.org/3/discover/movie?include_adult=false&include_video=false&language=en-US&page={page}&sort_by=vote_count.desc'

    r = requests.get(url, headers=auth_header)
    data = r.json()

    return [x['id'] for x in data['results']]

def get_popular_ids(start, end):
    seen_ids = {}
    for i in range(start, end + 1):
        new_ids = scrape_movie_page(i)
        for j in new_ids:
            seen_ids[j] = j

    with open(f'data/movies_pages_{start}_{end}.json', 'w') as f:
        f.writelines(json.dumps(seen_ids))

# load movies from a json file into the cache
# pages 1 - 50 already in movies directory
def fetch_from_json(json_file):
    with open(json_file, "r") as f:
        movie_dict = json.loads("".join(f.readlines()))
        count = 0

        for i in movie_dict:
            fetch_movie(i)
            count += 1

            if count % 50 == 0:
                print(f"{count} / {len(movie_dict)}")