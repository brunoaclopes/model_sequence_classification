from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizerFast
import tensorflow as tf


class Model:
    tokenizer = None
    model = None

    def __init__(self, model="distilbert-base-uncased", num_labels=2, learning_rate=1e-5):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model)
        self.model = TFDistilBertForSequenceClassification.from_pretrained(model, num_labels=num_labels)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def compile(self, metrics=None):
        if metrics is None:
            metrics = ['accuracy']
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=metrics)

    # def fit_model(self, dataset, epochs=3):
    #     self.model.fit(dataset, epochs)
    #
    # def evaluate_model(self, dataset):
    #     self.model.evaluate(dataset)

    def predict_proba(self, dataset):
        # predict
        predictions = self.model.predict(dataset).logits

        # transform to array with probabilities
        res = tf.nn.softmax(predictions, axis=1).numpy()

        return res

    def save(self, path="./model/pretrained"):
        self.model.save_pretrained(path)