from datasets import load_dataset
import tensorflow as tf


class Dataset:
    dataset = None

    def __init__(self, tokenizer, dataset="imdb"):
        self.tokenized_dataset = None
        self.dataset = load_dataset(dataset)
        self.tokenizer = tokenizer

    def tokenize(self, data_label="text"):
        def preprocess_function(examples):
            return self.tokenizer(examples[data_label], padding=True, truncation=True)

        self.tokenized_dataset = self.dataset.map(preprocess_function, batched=True, batch_size=None)

    def convert_to_tf(self, batch_size=16, shuffle=1000):
        self.tokenized_dataset.set_format('tf', columns=['input_ids', 'attention_mask', 'label'])

        def order(inp):
            """
            This function will group all the inputs of BERT
            into a single dictionary and then output it with
            labels.
            """
            data = list(inp.values())
            return {
                'input_ids': data[1],
                'attention_mask': data[2],
            }, data[0]

        processed_datasets = {}
        # Iterate over dataset entries
        for split_name, split_data in self.tokenized_dataset.items():
            # Convert split data to TensorFlow dataset format
            dataset = tf.data.Dataset.from_tensor_slices(split_data[:])
            # Set batch size and shuffle
            dataset = dataset.batch(batch_size).shuffle(shuffle)
            # Map the `order` function
            dataset = dataset.map(order, num_parallel_calls=tf.data.AUTOTUNE)

            # Store the processed dataset in the dictionary
            processed_datasets[split_name] = dataset

        return processed_datasets

    def convert_to_tf2(self, batch_size=16, shuffle=1000):
        self.tokenized_dataset.set_format('tf', columns=['input_ids', 'attention_mask', 'label'])

        def order(inp):
            """
            This function will group all the inputs of BERT
            into a single dictionary and then output it with
            labels.
            """
            data = list(inp.values())
            return {
                'input_ids': data[1],
                'attention_mask': data[2],
            }, data[0]

        processed_datasets = {}

        # converting train split of `emotions_encoded` to tensorflow format
        processed_datasets["train"] = tf.data.Dataset.from_tensor_slices(self.tokenized_dataset['train'][:])
        # set batch_size and shuffle
        processed_datasets["train"] = processed_datasets["train"].batch(batch_size).shuffle(shuffle)
        # map the `order` function
        processed_datasets["train"] = processed_datasets["train"].map(order, num_parallel_calls=tf.data.AUTOTUNE)

        # ... doing the same for test set ...
        processed_datasets["test"] = tf.data.Dataset.from_tensor_slices(self.tokenized_dataset['test'][:])
        processed_datasets["test"] = processed_datasets["test"].batch(batch_size)
        processed_datasets["test"] = processed_datasets["test"].map(order, num_parallel_calls=tf.data.AUTOTUNE)
        processed_datasets["unsupervised"] = tf.data.Dataset.from_tensor_slices(self.tokenized_dataset['unsupervised'][:])
        processed_datasets["unsupervised"] = processed_datasets["unsupervised"].batch(batch_size)
        processed_datasets["unsupervised"] = processed_datasets["unsupervised"].map(order, num_parallel_calls=tf.data.AUTOTUNE)

        return processed_datasets
