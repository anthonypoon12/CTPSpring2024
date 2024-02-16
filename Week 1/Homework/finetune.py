from transformers import Trainer, TrainingArguments

training_args = TrainingArguments("test-trainer")

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()