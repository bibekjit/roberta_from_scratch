from nltk.tokenize import sent_tokenize
from custom_tokenizer import BytePairEncodingTokenizer
import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path_to_raw_data", type=str, default="raw_data.pkl",
                    help="Path to the raw data file (default: raw_data.pkl)")
parser.add_argument("--maxlen", type=int, default=128,
                    help="Maximum sequence length (default: 128)")

args = parser.parse_args()

# Assign arguments to variables
PATH_TO_RAW_DATA = args.path_to_raw_data
MAXLEN = args.maxlen

# load raw data
with open(PATH_TO_RAW_DATA,'rb') as f:
    text = pickle.load(f)

# initalize and load tokenizer
tokenizer = BytePairEncodingTokenizer()
tokenizer.load_tokenizer('tokenizer.pkl')
tokenizer.maxlen = MAXLEN


def split_long_text(text, maxlen):
    """
    splitting long text sequences to max length
    :param text: list of string sequence
    :param maxlen: max sequence length after splitting
    :return: sequences split to maxlen
    """
    if len(text.split()) > maxlen:
        sents = sent_tokenize(text)
        sents = [s.replace('\n', '') for s in sents]
        output = []
        while True:

            tmp = []
            REM = False
            for i, s in enumerate(sents):
                s = s.split()[:maxlen]
                if len(tmp + s) <= maxlen:
                    tmp.extend(s)
                else:
                    REM = True
                    break

            tmp = ' '.join(tmp)

            output.append(tmp)

            if REM:
                sents = sents[i:]
            else:
                break
        return output

    else:
        return text


print('\nsplitting long text for pretraining data...')
seqs = []

for seq in text:
    x = split_long_text(seq,MAXLEN-2)
    if type(x) == list:
        seqs.extend(x)
    else:
        seqs.append(x)

# keep sequences with more than 3 tokens
seqs = [x for x in seqs if len(x) >= 3]

# tokenize and pad sequence
print('tokenizing sequences...')
seqs = [tokenizer.tokenize(x) for x in seqs]
seqs = [x for x in seqs if x.count(3)/len(x) <= 0.05]
print('padding sequences...')
seqs = np.asarray([tokenizer.add_padding(x) for x in seqs],dtype=np.int16)
np.random.shuffle(seqs)

# save tokenized data and updated tokenizer
print(f'pretraining data saved, number of sequences : {len(seqs)}')
with open('roberta_pretraining_data.pkl','wb') as f:
    pickle.dump(seqs,f)

tokenizer.save_tokenizer('tokenizer.pkl')
