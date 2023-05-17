# -*- coding: utf-8 -*-
"""
Created on Sun May 14 10:45:46 2023

@author: mbelic
"""
import random
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, Dataset

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Load and preprocess the dataset
def preprocess_function(examples):
    out = [tokenizer(example,
                     padding="max_length",
                     truncation=True,
                     max_length=512).input_ids for example in examples] 
    return out

go_emotions = load_dataset('go_emotions')

train_dataset = Dataset.from_dict({
    'input_ids': preprocess_function(go_emotions['train']['text']),
    'labels': preprocess_function(go_emotions['train']['text'])
})



valid_dataset = Dataset.from_dict({
    'input_ids': preprocess_function(go_emotions['validation']['text']),
    'labels': preprocess_function(go_emotions['validation']['text'])
})


training_args = Seq2SeqTrainingArguments(
    output_dir='./output',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_steps=100,
    logging_steps=20,
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_dir='./logs',
    seed=random.randint(0, 2**32-1),
    load_best_model_at_end=True,
    metric_for_best_model="loss",
)

# Train the model
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

trainer.train()
trainer.save_model('./fine_tuned_model')

