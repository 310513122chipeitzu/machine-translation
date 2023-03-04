from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,Seq2SeqTrainingArguments

import pandas as pd

from transformers import DataCollatorForSeq2Seq
import torch
from datasets import load_dataset,DatasetDict
# fetch model -> tokenizer and seq2seq model
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

# get data
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
data = load_dataset('csv', data_files={'train': './train.csv'})
# seperate data -> train eval test
train_test_dataset = data['train'].train_test_split(test_size=0.1)

test_valid = train_test_dataset['test'].train_test_split(test_size=0.5)

data = DatasetDict({
    'train': train_test_dataset['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})
    
# data preprocess
max_input_length = 128
max_target_length = 128
def preprocess_function(examples):
    inputs = examples['txt']
    targets = examples['ans']
    model_inputs = tokenizer(inputs,padding=True, max_length=max_input_length, truncation=True,return_tensors="pt")

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets,padding=True, max_length=max_target_length, truncation=True,return_tensors="pt")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

data = data.map(preprocess_function,batched=True)
data.set_format(type="torch")

# training args
batch_size = 16
args = Seq2SeqTrainingArguments(
    "test-translation",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=10,
    predict_with_generate=True,
    fp16=False,
)

from transformers import Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=data['train'],
    eval_dataset = data["valid"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
trainer.train()

# predict
test = pd.read_csv('./test-ZH-nospace.csv',usecols=['txt'])
ans = []
for i in range(len(test)):
    input_data = test.iloc[i][0]
    tmp = ''
    for k in input_data:
        # decode step
        # 1. tokenize words
        # 2. generate vectors via seq2seq model
        # 3. decode by tokenizer
        tokens= model.generate(tokenizer(k,truncation=True,return_tensors="pt").to('cuda')['input_ids'])
        output = tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)
        tmp += output
        tmp += ' '
    ans.append([tmp])
pd.DataFrame(ans).to_csv('./sub.csv',index=True)
tokens= model.generate(tokenizer(['拉'],truncation=True,return_tensors="pt").to('cuda')['input_ids'])
output = tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)
print(output)
Footer
© 2023 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
