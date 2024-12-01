from custom_tokenizer import BytePairEncodingTokenizer
import pickle
import time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--path_to_raw_data", type=str, default="raw_data.pkl",
                    help="Path to the raw data file (default: raw_data.pkl)")
parser.add_argument("--new_special_tokens", type=str, nargs='+', default=['USER', 'LINK'],
                    help="List of new special tokens to add (default: ['USER', 'LINK'])")
parser.add_argument("--num_iterations", type=int, default=5,
                    help="Number of iterations for training (default: 5)")
parser.add_argument("--min_pair_freq", type=int, default=50,
                    help="Minimum frequency of token pairs (default: 50)")
parser.add_argument("--num_tokens", type=int, default=20000,
                    help="Number of tokens in the vocabulary (default: 20000)")
parser.add_argument("--max_word_count", type=int, default=20000 * 10,
                    help="Maximum word count for tokenization (default: 200000)")

args = parser.parse_args()

# Assign arguments to variables
PATH_TO_RAW_DATA = args.path_to_raw_data
NEW_SPECIAL_TOKENS = args.new_special_tokens
NUM_ITERATIONS = args.num_iterations
MIN_PAIR_FREQ = args.min_pair_freq
NUM_TOKENS = args.num_tokens
MAX_WORD_COUNT = args.max_word_count


with open(PATH_TO_RAW_DATA,'rb') as f:
    text = pickle.load(f)

# initialize wordpiece tokenizer
tokenizer = BytePairEncodingTokenizer(num_tokens=NUM_TOKENS,
                                      max_word_count=MAX_WORD_COUNT)


# create corpus
print('creating corpus...')
tokenizer(text)

start = time.time()


# train tokenizer

# data also include twitter handles and URLs which are normalized as USER and LINK
# they are added as special tokens and not trained upon

print('tokenizer training in progress...')
tokenizer.train(iterations=NUM_ITERATIONS,
                min_pair_freq=MIN_PAIR_FREQ,
                special_tokens=NEW_SPECIAL_TOKENS)

end = time.time()
tokenizer.save_tokenizer('tokenizer.pkl')
print(f'training time : {round((end - start) / 60, 2)} minutes')





