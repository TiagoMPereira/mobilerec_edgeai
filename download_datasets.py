from datasets import load_dataset
import pandas as pd

# load the dataset and metadata
metadata = load_dataset('recmeapp/mobilerec', data_dir='app_meta')
metadata["train"].to_csv('./data/input/metadata.csv')

print("Metadata downloaded")

interactions = load_dataset('recmeapp/mobilerec', data_dir='interactions')
interactions["train"].to_csv('./data/input/interactions.csv')

print("Interactions downloaded")