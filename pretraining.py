import tensorflow as tf
import numpy as np
from custom_tokenizer import BytePairEncodingTokenizer
from lr_scheduler import CosineLRScheduler
from tqdm.auto import tqdm
import json
import pickle
from bert import BERT
from nltk.corpus import stopwords
from string import punctuation
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--val_split_percent", type=float, default=0.1,
                    help="Validation split percentage (default: 0.1)")
parser.add_argument("--n_heads", type=int, default=8,
                    help="Number of attention heads (default: 8)")
parser.add_argument("--d_model", type=int, default=256,
                    help="Dimension of the model (default: 256)")
parser.add_argument("--n_layers", type=int, default=2,
                    help="Number of transformer layers (default: 2)")
parser.add_argument("--units", type=int, default=256 * 4,
                    help="Feed-forward network size (default: 1024)")
parser.add_argument("--batch", type=int, default=64,
                    help="Batch size (default: 64)")
parser.add_argument("--masking_prob", type=float, default=0.15,
                    help="Masking probability for dynamic masking (default: 0.15)")
parser.add_argument("--init_lr", type=float, default=1e-7,
                    help="Initial learning rate (default: 1e-6)")
parser.add_argument("--peak_lr", type=float, default=3e-4,
                    help="Peak learning rate (default: 3e-4)")
parser.add_argument("--end_lr", type=float, default=1e-7,
                    help="Ending learning rate (default: 1e-6)")
parser.add_argument("--warmup_rate", type=float, default=0.06,
                    help="Warmup rate for learning rate (default: 0.06)")
parser.add_argument("--epochs", type=int, default=15,
                    help="Number of epochs (default: 15)")

args = parser.parse_args()

# Assign arguments to variables
VAL_SPLIT_PERCENT = args.val_split_percent
N_HEADS = args.n_heads
D_MODEL = args.d_model
N_LAYERS = args.n_layers
UNITS = args.units
BATCH = args.batch
MASKING_PROB = args.masking_prob
INIT_LR = args.init_lr
PEAK_LR = args.peak_lr
END_LR = args.end_lr
WARMUP_RATE = args.warmup_rate
EPOCHS = args.epochs

# load trained tokenizer
tokenizer = BytePairEncodingTokenizer()
tokenizer.load_tokenizer('tokenizer.pkl')

# load the data
with open('roberta_pretraining_data.pkl','rb') as f:
    data = pickle.load(f)

# set maxlen and vocab_size
tokenizer.maxlen = data.shape[1]
maxlen = data.shape[1]
vocab = len(tokenizer.i2w)

# split data
val_data = data[:int(len(data) * VAL_SPLIT_PERCENT)]
train_data = data[int(len(data) * VAL_SPLIT_PERCENT):]

# create model
input_layer = tf.keras.layers.Input((maxlen,),batch_size=BATCH,dtype=tf.int16)
bert_layer = BERT(D_MODEL,N_HEADS,vocab,maxlen,UNITS,N_LAYERS)
mlm_head = tf.keras.layers.Dense(vocab,activation='softmax')

x = bert_layer(input_layer)
output = mlm_head(x)

model = tf.keras.Model(input_layer,output)


print()
model.summary()
print()

# save model config
config = model.layers[1].get_config()

with open('roberta_config.json','w') as f:
    json.dump(config,f)

# train position embeddings
model.layers[1].embedding.pos_emb_layer.trainable = True

# initialize loss fucntion and optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.AdamW(weight_decay=0.01,
                                      beta_1=0.9,
                                      beta_2=0.98,
                                      epsilon=1e-6)


training_steps = len(train_data) // BATCH
val_steps = len(val_data) // BATCH

lr_scheduler = CosineLRScheduler(training_steps,PEAK_LR,END_LR,INIT_LR,WARMUP_RATE)
mask_token_idx = tokenizer.w2i['<mask>']

@tf.function
def train_step(x,y):
    with tf.GradientTape() as tape:
        pred = model(x,training=True)
        loss = loss_fn(y,pred,sample_weight=x == mask_token_idx)
    weights = model.trainable_weights
    grads = tape.gradient(loss,weights)
    optimizer.apply_gradients(zip(grads,weights))
    return loss

@tf.function
def test_step(x,y):
    pred = model(x,training=False)
    loss = loss_fn(y,pred,sample_weight=x == mask_token_idx)
    return loss

def create_train_batches(data, total_steps, batch):
    """
    function to create batch and shuffle pretraining data
    :param data: 2d int array
    :param total_steps: total number of training steps
    :param batch: batch size
    :return: 3d int array
    """
    input_len = data.shape[-1]
    train = np.random.permutation(data)
    train_batches = train[:total_steps * batch,:].copy().reshape((total_steps,batch,input_len))
    return train_batches

# create train and val batches
train_batches = create_train_batches(train_data, training_steps, BATCH)
val_batches = val_data[:val_steps * BATCH,:].copy().reshape((val_steps, BATCH, maxlen))

# do not mask most frequent tokens
# these tokens will not be masked half of the batches
stopwords = stopwords.words('english')
punct = [p for p in punctuation if p in tokenizer.w2i]
dont_mask = ['<pad>','<cls>','<sep>','<unk>'] + list(tokenizer.words)[:25] + punct
dont_mask = np.array([tokenizer.w2i[w] for w in dont_mask],np.int16)


def dynamic_masking(x, prob=0.2, mask_token=4, dont_mask=np.array([0, 1, 2, 3])):
    """
    dynamically mask tokens in sequence
    :param x: 2d int array
    :param prob: masking probability 
    :param mask_token: masking token value
    :param dont_mask: tokens not be masked
    :return: masked sequence (2d int array)
    """
    not_mask = np.array(~np.isin(x, dont_mask), np.int16)

    n_to_mask = (np.sum(not_mask, axis=1) * prob).astype('int')

    full_mask = np.where(n_to_mask == 0, 1, 0)
    full_mask = full_mask[:, np.newaxis] * not_mask
    full_mask = full_mask + not_mask
    full_mask = full_mask == 2
    x = np.where(full_mask, mask_token, x)

    not_mask = np.array(~np.isin(x, dont_mask), np.int16)

    mask = not_mask.copy()
    batch, maxlen = mask.shape
    arr = np.array(np.linspace(0, maxlen - 1, maxlen), np.int16)
    arr = np.tile(arr, batch).reshape(mask.shape)
    arr = arr * mask

    for i, row in enumerate(arr):
        if n_to_mask[i] > 0:
            ids = np.where(row != 0)[0]
            ids = np.random.permutation(ids)
            ids = ids[:n_to_mask[i]]
            mask[i][ids] = 0

    not_mask = not_mask + mask
    mask = not_mask == 1

    x = np.where(mask, mask_token, x)
    return x


val_mask = [dynamic_masking(x,prob=MASKING_PROB) for x in val_batches]
val_mask = np.array(val_mask,np.int16)

print("\n=================================================================\n")

epochs = EPOCHS

losses = {'train':[],'valid':[]}

for e in range(1, epochs + 1):

    print(f'EPOCH : {e}/{epochs}')

    loss = 0
    for i in tqdm(range(training_steps)):

        x = train_batches[i, :, :]
        if i % 2 != 0:
            x_mask = dynamic_masking(x, prob=MASKING_PROB)
        else:
            x_mask = dynamic_masking(x, prob=MASKING_PROB, dont_mask=dont_mask)
        lr = lr_scheduler(i + 1)
        optimizer.learning_rate.assign(lr)
        loss += train_step(x_mask, x)

    train_loss = round(loss.numpy() / (i + 1), 4)

    losses['train'].append(train_loss)

    loss = 0
    for i in tqdm(range(val_steps)):
        x = val_batches[i, :, :]
        x_mask = val_mask[i, :, :]
        loss += test_step(x_mask, x)

    loss = round(loss.numpy() / (i + 1), 4)

    print('train_loss :', train_loss)
    print('val_loss :', loss)
    print('============================')

    if e == 1:
        model.save_weights(f'checkpoints/mlm_weights_{e}.weights.h5')
        model.save_weights('mlm_pretraining.weights.h5')

        weights = model.layers[1].get_weights()
        with open('roberta_weights.pkl','wb') as f:
            pickle.dump(weights,f)

        print('weights saved\n')

    elif e > 1 and min(losses['valid']) > loss:
        model.save_weights(f'checkpoints/mlm_weights_{e}.weights.h5')
        model.save_weights('mlm_pretraining.weights.h5')

        weights = model.layers[1].get_weights()
        with open('roberta_weights.pkl', 'wb') as f:
            pickle.dump(weights, f)

        print('weights saved\n')

    losses['valid'].append(loss)

    print()

    inp = val_mask[0][0]
    idx = np.where(inp == 4)[0]
    real = val_batches[0][0]
    inp = inp[np.newaxis,:]
    pred = model(inp)
    inp_loss = loss_fn(real[np.newaxis,:],pred)
    pred = np.argmax(pred[0],axis=-1)
    print('predicted tokens :',[tokenizer.i2w[t] for t in pred[idx]])
    print('actual tokens :', [tokenizer.i2w[t] for t in real[idx]])
    print('sample loss :',round(inp_loss.numpy()/len(idx),4))

    train_batches = create_train_batches(train_data, training_steps, BATCH)

    with open('training_logs.json','w') as f:
        json.dump(losses,f)


model.save('mlm_model.keras')

print('--------PRETRAINING COMPLETED---------')

