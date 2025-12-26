import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from util.embeddings import ENERGY_WORDS, HUMOR_WORDS, WARMTH_WORDS, TENSION_WORDS, DARKNESS_WORDS

sentiment_model_path = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path)

embeddings = SentenceTransformer("all-MiniLM-L6-v2")
darkness = (embeddings.encode(DARKNESS_WORDS['pos']), embeddings.encode(DARKNESS_WORDS['neg']))
tension = (embeddings.encode(TENSION_WORDS['pos']), embeddings.encode(TENSION_WORDS['neg']))
warmth = (embeddings.encode(WARMTH_WORDS['pos']), embeddings.encode(WARMTH_WORDS['neg']))
energy = (embeddings.encode(ENERGY_WORDS['pos']), embeddings.encode(ENERGY_WORDS['neg']))
humor = (embeddings.encode(HUMOR_WORDS['pos']), embeddings.encode(HUMOR_WORDS['neg']))

dimensions = [energy, darkness, tension, warmth, humor]

def run_embeddings(chunked):
    encoded = [embeddings.encode(x) for x in chunked]
    result = []
    for i in dimensions:
        pos_tensor = embeddings.similarity(encoded, i[0])
        neg_tensor = embeddings.similarity(encoded, i[1])

        k = min(len(encoded), 2)
        # take 3 largest values to eliminate background noise from plot
        top_val = (torch.mean(torch.topk(pos_tensor, k=k, dim=0)[0])
                   - torch.mean(torch.topk(neg_tensor, k=k, dim=0)[0]))
        result.append(top_val.item())

    return result

def get_valence(chunked):
    encoded_input = [tokenizer(x, return_tensors='pt') for x in chunked]
    valence = []
    for i in encoded_input:
        output = sentiment_model(**i)
        scores = output.logits
        scores = torch.softmax(scores, dim=1)[0]

        neg, neu, pos = scores.tolist()
        valence.append(pos - neg)

    return sum(valence) / len(valence)