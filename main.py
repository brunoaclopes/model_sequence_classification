from dataset.dataset import Dataset
from model.model import Model

import tensorflow as tf
import os

print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))

if os.path.exists("./model/pretrained"):
    model_name = "./model/pretrained"
else:
    model_name = "distilbert-base-uncased"


model = Model(model_name)

dataset = Dataset(model.tokenizer)
dataset.tokenize()
datasets = dataset.convert_to_tf2(batch_size=128)

model.compile()

# predicted = model.predict_proba(datasets["unsupervised"])
# print(predicted)

model.model.fit(datasets["train"])
model.save("./model/pretrained")

# model evaluation on the test set
model.model.evaluate(datasets['test'])

predicted = model.predict_proba(datasets["unsupervised"])

print(predicted)
