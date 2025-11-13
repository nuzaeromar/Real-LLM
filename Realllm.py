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
import datasets # use the Hugging Face's datasets library
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

from bertopic import BERTopic
from hdbscan import HDBSCAN
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer


def dataset_mapping(x):
    # Try to find the input text field
    for field in ["sentence", "text", "content", "review", "document"]:
        if field in x:
            return {
                "x": x[field],
                "y": 1 if x["label"] > 0.5 else 0,
            }
    raise ValueError(f"No valid text field found in example: {x}")

def snli_mapping(x):
    if x["label"] == -1:  # SNLI sometimes has -1 for invalid labels
        return None
    return {
        "x": f"{x['premise']} [SEP] {x['hypothesis']}",
        "y": x["label"]
    }

def map_multiclass(x):
    # AG News (and any other multi‐way task) keep the integer label as is
    text = x.get("sentence", x.get("text"))
    return {"x": text, "y": int(x["label"])}

def load_victim(model_name, dataset):
    victim = None
    if dataset == "sst":
        if model_name == "bert":
            victim = oa.DataManager.loadVictim("BERT.SST")
        elif model_name == "roberta":
            victim = oa.DataManager.loadVictim("ROBERTA.SST")
        else:
            raise ValueError(f"Unsupported model '{model_name}' for SST")
    else:
        if dataset == "ag_news":
            if model_name == "bert":
                model_id = "textattack/bert-base-uncased-ag-news"
            elif model_name == "roberta":
                model_id = "textattack/roberta-base-ag-news"

        elif dataset == "imdb":
            if model_name == "bert":
                model_id = "textattack/bert-base-uncased-imdb"
            elif model_name == "roberta":
                model_id = "textattack/roberta-base-imdb"

        elif dataset == "mr":
            if model_name == "bert":
                model_id = "textattack/bert-base-uncased-rotten-tomatoes"
            elif model_name == "roberta":
                model_id = "textattack/roberta-base-rotten-tomatoes"

        elif dataset == "yelp":
            if model_name == "bert":
                model_id = "textattack/bert-base-uncased-yelp-polarity"
            elif model_name == "roberta":
                model_id = "textattack/roberta-base-yelp-polarity"

        elif dataset in ["snli", "mnli", "mnli_matched", "mnli_mismatched"]:
            if model_name == "bert":
                model_id = "textattack/bert-base-uncased-snli" if dataset == "snli" else "textattack/bert-base-uncased-MNLI"
            elif model_name == "roberta":
                model_id = "textattack/roberta-base-snli" if dataset == "snli" else "textattack/roberta-base-MNLI"

        else:
            raise ValueError(f"No pre-trained model found for dataset '{dataset}'")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        word_embeddings = model.bert.embeddings.word_embeddings if model_name == "bert" else model.roberta.embeddings.word_embeddings
        victim = oa.classifiers.TransformersClassifier(model, tokenizer, word_embeddings)

    return victim


seed = 777
N = 100

results = {}

models = ['bert', 'roberta']
# models = ['bert']

dataset_names = ['sst','ag_news','snli']
# dataset_names = ['sst', 'ag_news','yelp', 'mr', 'snli', 'mnli_matched', 'mnli_mismatched']
# dataset_names = ['yelp', 'ag_news', 'snli',]


dataset_vault = {}
for dataset_name in dataset_names:

    if dataset_name == "yelp":
        dataset = datasets.load_dataset("yelp_polarity", split="test")
        mapper = dataset_mapping

    elif dataset_name == "mr":
        dataset = datasets.load_dataset("rotten_tomatoes", split="test")
        mapper = dataset_mapping

    elif dataset_name == "snli":
        dataset = datasets.load_dataset(dataset_name, split="validation") 
        mapper = snli_mapping
    
    elif dataset_name == "mnli_matched":
        dataset = datasets.load_dataset("multi_nli", split="validation_matched")
        mapper = snli_mapping
    elif dataset_name == "mnli_mismatched":
        dataset = datasets.load_dataset("multi_nli", split="validation_mismatched")
        mapper = snli_mapping
    else:
        dataset = datasets.load_dataset(dataset_name, split="test", trust_remote_code=True)
        mapper = dataset_mapping if dataset_name in ["sst", "imdb"] else map_multiclass


    dataset=dataset.map(function=mapper)
    dataset = dataset.filter(lambda x: x is not None)
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.select(list(range(min(N, len(dataset)))))
    dataset_vault[dataset_name] = dataset

all_label_words = {}  
for ds_name, ds in dataset_vault.items():
    ds_label_words = {}
    labels = sorted(set(ds["y"]))

    for lbl in labels:
        # a) Filter to only examples of this label
        ds_lbl = ds.filter(lambda ex, lab=lbl: ex["y"] == lab)
        # texts  = ds_lbl["x"]
        texts = [x.split("[SEP]")[-1].strip() for x in ds_lbl["x"]] if ds_name in ["snli", "mnli"] else ds_lbl["x"]


        # b) Fit BERTopic on this slice
        vectorizer_model = CountVectorizer(ngram_range=(1,1), stop_words="english")
        hdb = HDBSCAN(min_cluster_size=5, min_samples=1, prediction_data=True)
        model = BERTopic(
            vectorizer_model=vectorizer_model,
            hdbscan_model=hdb,
            language="english",
            calculate_probabilities=True,
            verbose=False
        )
        topics, probs = model.fit_transform(texts)

        # c) Get topic frequencies and pick top-20 (exclude outlier = -1)
        freq = model.get_topic_freq()        
        freq = freq[freq.Topic != -1]        
        top_topics = freq.nlargest(20, 'Count')['Topic'].tolist()

        # d) pool word‐score pairs from those topics
        agg_scores = defaultdict(float)
        for tid in top_topics:
            for word, score in model.get_topic(tid):
                agg_scores[word] += score

        # e) select the top 20 words by total score
        top20 = [
            w for w,_ in 
            sorted(agg_scores.items(), key=lambda kv: kv[1], reverse=True)[:20]
        ]
        ds_label_words[lbl] = top20
    all_label_words[ds_name] = ds_label_words


attacker_name = "BERTAttacker"

for model_name in models:
    for dataset_name in dataset_names:
        key = "{}-{}-{}".format(attacker_name, model_name, dataset_name)
        if key not in results:
            print(key)
            victim = load_victim(model_name, dataset_name)
            dataset = dataset_vault[dataset_name]
            bank     = all_label_words[dataset_name]  
            attacker = oa.attackers.BERTAttacker(label_words=bank)
            attack_eval = oa.AttackEval(attacker, victim, metrics = [
                            oa.metric.EditDistance(),
                            oa.metric.ModificationRate()
                        ])
            result = attack_eval.eval(dataset, all_label_words, visualize=True)

            if result:
                print(f"Attack Success Rate: {result['Attack Success Rate']}")
                print(f"Modification Rate: {result['Avg. Word Modif. Rate']}")

            # print(key, result)
            results[key] = {
                'Attack Success Rate': result['Attack Success Rate'],
                'Avg. Running Time': result['Avg. Running Time'],
                'Total Attacked Instances': result['Total Attacked Instances'],
                'Successful Instances': result['Successful Instances'],
                'Total Query Exceeded': result['Total Query Exceeded'],
                'Avg. Victim Model Queries': result['Avg. Victim Model Queries'],
                'Avg. Levenshtein Edit Distance': result['Avg. Levenshtein Edit Distance'],
                'Avg. Word Modif. Rate': result['Avg. Word Modif. Rate']
            }

            with open('./all_results_with_adversarials.pkl', 'wb') as f:
                pickle.dump(results, f)

            del victim

## Summarize results
df_data = []
metrics = [
    'Attack Success Rate', 
    'Avg. Running Time', 
    'Total Attacked Instances', 
    'Successful Instances', 
    'Total Query Exceeded', 
    'Avg. Victim Model Queries', 
    'Avg. Levenshtein Edit Distance', 
    'Avg. Word Modif. Rate'
]

# Load saved results
with open('./all_results_with_adversarials.pkl', 'rb') as f:
    results = pickle.load(f)

for model_name in models:
    for dataset_name in dataset_names:
        key = "{}-{}-{}".format(attacker_name, model_name, dataset_name)
        if key in results:
            result = results[key]
            tmp = {}  # Initialize tmp here, without attacker
            for metric in metrics:
                tmp[f'{model_name}-{dataset_name}-{metric}'] = result.get(metric, None)

        df_data.append(tmp)  # Move this line inside the `if`


# Convert to DataFrame
df = pd.DataFrame.from_dict(df_data)
print(df)


