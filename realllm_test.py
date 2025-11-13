import ssl
import urllib.request
import pickle

# Create an “unverified” SSL context
ctx = ssl._create_unverified_context()

opener = urllib.request.build_opener(
    urllib.request.HTTPSHandler(context=ctx)
)
urllib.request.install_opener(opener)
import OpenAttack as oa
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 1. Load victim model for MR
model_id = "textattack/bert-base-uncased-rotten-tomatoes"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
word_embeddings = model.bert.embeddings.word_embeddings
victim = oa.classifiers.TransformersClassifier(model, tokenizer, word_embeddings)

from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from datasets import load_dataset

# Load MR dataset (Rotten Tomatoes)
mr_dataset = load_dataset("rotten_tomatoes", split="train")

# We'll take 100 samples from each label for topic modeling
label_words = {}
for label in [0, 1]:
    # Filter samples of this label
    ds_lbl = mr_dataset.filter(lambda x: x['label'] == label).shuffle(seed=42).select(range(100))
    texts = ds_lbl["text"]

    # Fit BERTopic
    vectorizer_model = CountVectorizer(ngram_range=(1, 1), stop_words="english")
    hdb = HDBSCAN(min_cluster_size=5, min_samples=1, prediction_data=True)
    model = BERTopic(
        vectorizer_model=vectorizer_model,
        hdbscan_model=hdb,
        language="english",
        calculate_probabilities=True,
        verbose=False
    )
    topics, probs = model.fit_transform(texts)

    # Get top topic words
    freq = model.get_topic_freq()
    freq = freq[freq.Topic != -1]
    top_topics = freq.nlargest(30, 'Count')['Topic'].tolist()

    # Aggregate word scores
    agg_scores = defaultdict(float)
    for tid in top_topics:
        for word, score in model.get_topic(tid):
            agg_scores[word] += score

    # Take top 20
    top_words = sorted(agg_scores.items(), key=lambda kv: kv[1], reverse=True)
    label_words[label] = [w for w, _ in top_words[:20]]

# print(label_words)

attacker_name = "BERTAttacker"
models = ['bert']
results = {}


# 2. Define the input
sample_text = "Davis is so enamored of her own creation that she can not see how insufferable the character is."
sample_label = 0  # change to 1 if you want to test positive sentiment

# Run attack using attacker with topic-based label words
attacker = oa.attackers.BERTAttacker(label_words=label_words)

# Wrap for OpenAttack eval (optional)
dataset = [{"x": sample_text, "y": sample_label}]
attack_eval = oa.AttackEval(attacker, victim, metrics=[
    oa.metric.EditDistance(),
    oa.metric.ModificationRate()
])
result = attack_eval.eval(dataset, visualize=True)

print("Result:", result)
