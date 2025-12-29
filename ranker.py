import faiss
import numpy as np
import pandas as pd

weight = np.array([2, 0.75, 1, 1, 1, 0.75])

class MovieRanker:
    def __init__(self):
        # load z score table
        df = pd.read_csv("data/movies-z.csv")
        self.df = df

        # only look at vector columns
        cols = ['valence', 'energy', 'darkness', 'tension', 'warmth', 'humor']
        movie_vectors = df[cols].values.astype("float32")

        # normalize movie vector in preparation of indexing
        movie_norms = np.linalg.norm(movie_vectors, axis=1, keepdims=True)
        movie_vec_normal = movie_vectors / movie_norms

        self.movie_norms = movie_norms

        # index normal vectors using faiss for cosine
        d = movie_vec_normal.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(movie_vec_normal)

        self.cosine = index

        # then index raw vectors for L2
        index = faiss.IndexFlatL2(d)
        index.add(movie_vectors)
        self.L2 = index

    # return top k movie titles given a z scored song vector
    def top_k_movies(self, song_vector, k=5):
        # normalization
        song_vec = np.array(song_vector).astype("float32") * weight
        song_norm = np.linalg.norm(song_vec)
        song_vec_normal = song_vec / song_norm

        # faiss expects 2d vector
        song_unit = song_vec_normal.reshape(1, -1)
        song_vec = song_vec.reshape(1, -1)

        # search index - narrow it down to 100 movies
        cosine, cos_indices = self.cosine.search(song_unit, 100)
        l2, l2_indices = self.L2.search(song_vec, 100)

        # intersection of the two searches
        intersection = np.intersect1d(cos_indices[0], l2_indices[0])

        # map movie ids to scores
        score_by_movie = {}

        for movie_id in intersection:
            # get cosine and L2 score
            cos_idx = np.where(cos_indices[0] == movie_id)[0][0] + 1
            l2_idx = np.where(l2_indices[0] == movie_id)[0][0] + 1

            # add to map
            # scores - cosine[0][cos_idx] and l2_norm[0][l2_idx]
            score_by_movie[movie_id] = (
                    0.5 * cos_idx +
                    0.5 * l2_idx
            )

        # limit to k and sort based on score
        ranked = sorted(
            score_by_movie.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        movie_ids = [mid for mid, _ in ranked]
        scores = [s for _, s in ranked]

        titles = self.df.loc[movie_ids]['movie'].tolist()
        urls = self.df.loc[movie_ids]['url'].tolist()

        # return movie titles
        return titles, scores, urls
