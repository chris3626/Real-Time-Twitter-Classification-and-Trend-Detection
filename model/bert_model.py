from transformers import BertTokenizerFast, TFDistilBertForSequenceClassification
from transformers import DataCollatorWithPadding, create_optimizer
from datasets import Dataset
import tensorflow as tf
from sklearn.metrics import classification_report
import os

def prepare_data_for_bert(data_etl):
    data_bert = data_etl[['text', 'target']]
    dataset_bert = Dataset.from_pandas(data_bert)
    dataset_bert = dataset_bert.train_test_split(test_size=0.2)

    tokenizer = BertTokenizerFast.from_pretrained("distilbert-base-uncased")
    def preprocess(example):
        return tokenizer(example['text'], padding="max_length", truncation=True, max_length=64)
    
    tokenized = dataset_bert.map(preprocess, batched=True, batch_size=1000)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

    train_ds = tokenized["train"].to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols=["target"],
        shuffle=True,
        batch_size=8,
        collate_fn=collator
    )

    test_ds = tokenized["test"].remove_columns(["target"]).to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        shuffle=False,
        batch_size=8,
        collate_fn=collator
    )

    return dataset_bert, tokenized, train_ds, test_ds, tokenizer

def fine_tune_bert(train_ds, tokenizer, tokenized, epochs=1):
    train_steps = len(tokenized["train"]) // 8 * epochs
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        optimizer, _ = create_optimizer(2e-5, 0, train_steps)

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )

        model.fit(train_ds, epochs=epochs)
        model.save_pretrained("bert_sentiment_model")
        tokenizer.save_pretrained("bert_sentiment_model")
    
    return model

def evaluate_bert_model(test_ds, dataset_bert):
    model = TFDistilBertForSequenceClassification.from_pretrained("bert_sentiment_model")
    tokenizer = BertTokenizerFast.from_pretrained("bert_sentiment_model")

    logits = model.predict(test_ds).logits
    predictions = tf.argmax(logits, axis=1).numpy()
    y_true = dataset_bert["test"]["target"]

    print(classification_report(y_true, predictions))

def run_bert_pipeline(data_etl, train_new_model=False):
    dataset_bert, tokenized, train_ds, test_ds, tokenizer = prepare_data_for_bert(data_etl)

    if train_new_model or not os.path.exists("bert_sentiment_model"):
        fine_tune_bert(train_ds, tokenizer, tokenized)

    evaluate_bert_model(test_ds, dataset_bert)
