## ROBERTA Implementation

### model config

`L=2` `H=256` `A=8` `maxlen=128` `ff_units=1024` 

15% dynamic masking (whole sequence is masked if sequence is very small)

### tokenizer training config

tokenizer trained on 200k unique tokens

vocabulary size = 20020

minimum pair frequency = 50
training iterations = 5






