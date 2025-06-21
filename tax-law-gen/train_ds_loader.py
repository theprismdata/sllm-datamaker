import json

data_list = []
with open("training_dataset_example.jsonl", "r") as f:
    for line in f:
        data_list.append( json.loads(line))
for data in data_list:
    print(data)
    