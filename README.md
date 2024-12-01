## ROBERTA Implementation

### model config

`L=2` `H=256` `A=8` `maxlen=128` `ff_units=1024` 

15% dynamic masking (whole sequence is masked if sequence is very small)

### Byte Pair tokenizer training config 

tokenizer trained on 200k unique tokens

vocabulary size = 20020

minimum pair frequency = 50

training iterations = 5


### Adam optimizer config 

`peak_lr=3e-4` `initial_lr=1e-7` `end_lr=1e-7` 

`weight_decay=0.01` `betas=0.9 and 0.98` `epsilon=1e-6`

### Model pretraining

Model pretrained on total 270k plus steps (10 epochs * 27k steps per epoch)

Pretraining was done on RTX 2060 mobile and took 150hrs to pretrain

### Trained Postional Embedding
![trained positional embeddings](https://github.com/user-attachments/assets/5812f2a5-c736-4cf8-b848-66bd0b705008)

### Loss Logs
![losses](https://github.com/user-attachments/assets/59043b82-ded9-4c22-8ff7-a3f2ce4e1ca7)









