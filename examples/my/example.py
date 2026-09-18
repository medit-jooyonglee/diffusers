# import torch


from datasets import load_dataset

path = '/workspace/dataset/nsfw/nude/'
dataset = load_dataset(
    "imagefolder",
    data_dir=path
)

print(dataset)
print(dataset["train"][0])
