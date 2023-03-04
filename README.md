# machine-translation
# machine-translation-310513122chipeitzu
machine-translation-310513122chipeitzu created by GitHub Classroom

## 使用Hugging Face已經訓練好的 NLP MODEL
![image](https://user-images.githubusercontent.com/114553468/212009198-2eb7a422-5c8d-4f7c-a68a-ab34a1ad397b.png)

## pip
###### 注意安裝順序(transformers需要比sentencepiece先安裝好)
```
!pip  install transformers
!pip install datasets
!pip install sentencepiece
```

## fetch model -> tokenizer and seq2seq model
```
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
```
![image](https://user-images.githubusercontent.com/114553468/211820343-30f68a63-142f-4384-8b81-297c2fa08cb2.png)

## get data
```
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
data = load_dataset('csv', data_files={'train': './train.csv'})
```

## seperate data -> train eval test
```
train_test_dataset = data['train'].train_test_split(test_size=0.1)

test_valid = train_test_dataset['test'].train_test_split(test_size=0.5)

data = DatasetDict({
    'train': train_test_dataset['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})
```
## data preprocess
```
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
```
## training args
```
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
```
## predict
```
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
```
## grade
![image](https://user-images.githubusercontent.com/114553468/211821380-1030e26b-7e93-474c-8c03-fa352e86cc33.png)
