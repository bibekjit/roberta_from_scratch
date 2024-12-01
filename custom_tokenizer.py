from collections import Counter
import re
import numpy as np
import string
import pickle


class WordPieceTokenizer:
    """
    BERT's WordPiece tokenizer implementation

    Tokenizer is to be trained on a large corpus and learns the
    sequences of letters in all the words it is trained upon

    the learned subwords are stored as prefixes and suffixes.
    Suffixes are identified with '##' notation

    example: playing -> play ##ing

    """
    def __init__(self,num_tokens,maxlen=None,max_word_count=None):

        """
        :param num_tokens: number tokens within the vocab (this is excluding special tokens)
        :param maxlen: maximum sequence length to keep, rest will be truncated
        :param max_word_count: maximum number of tokens for tokenizer training.
        If it is set to None, `num_tokens` will be considered for training.
        Otherwise, should be greater than `num_tokens`
        """
        self.special_toks = ['<pad>', '<cls>', '<sep>', '<unk>', '<mask>']
        self.w2i = {x:i for i,x in enumerate(self.special_toks)}
        self.i2w = {self.w2i[k]: k for k in self.w2i}
        self.size = len(self.special_toks)
        self.tokens = {}
        self.words = {}
        self.vocab = {}
        self.subwords = {}
        self.maxlen = maxlen
        self.num_tokens = num_tokens
        self.prefix = []
        self.suffix = []
        self.max_word_count = max_word_count

        if max_word_count is not None:
            assert max_word_count > num_tokens

    def __call__(self, text):

        """
        function to create corpus

        :param text: string sequences within lists
        """

        # gather the top k most frequent words along with their frequency

        corpus = ' '.join(text).split()
        tok_freq = Counter(corpus)
        tokens = sorted(corpus, key=lambda x: (tok_freq[x], x), reverse=True)

        count = 1

        for i in range(1, len(tokens)):

            if tokens[i] == tokens[i - 1]:
                count += 1
            else:
                self.tokens[tokens[i - 1]] = count

                if tokens[i - 1].isalnum():
                    self.words[tokens[i - 1]] = count

                count = 1

            if len(self.words) == self.num_tokens and self.max_word_count is None:
                break

            elif self.max_word_count is not None and len(self.words) == self.max_word_count:
                break

    def train(self, iterations=1, min_pair_freq=10,
              strategy='coverage', special_tokens=[]):

        """
        Tokenizer training function
        The tokenizer learns the subword sequences and keeps the once as per
        their frequency threshold

        The subword sequences are segregated as prefix or suffix

        :param iterations: number of training iterations
        :param min_pair_freq: minimum subword pair frequency
        :param strategy: strategy for creating subwords. Frequency-based will create subwords
         based on frequency of the corpus and coverage-based will create based on number unique
         tokens subword is found in.
        :param special_tokens: list of new special tokens to be added
        """

        # split into subwords

        if strategy == 'coverage':
            word_freq = {k:1 for k in self.words}

        elif strategy == 'frequency':
            word_freq = self.words.copy()

        else:
            print(f'error : `{strategy}` is not valid')
            return

        if len(special_tokens) > 0:
            for tok in special_tokens:
                if tok in word_freq:
                    del word_freq[tok]

        subwords = [w.replace('', '_')[1:-1] for w in word_freq]

        # get the most frequent subword pairs
        for _ in range(iterations):
            combs = {}
            for i, w in enumerate(subwords):
                w = w.split('_')
                for j, _ in enumerate(w):
                    count = word_freq[subwords[i].replace('_', '')]
                    comb = '_'.join(w[j:j + 2])
                    if comb in combs:
                        combs[comb] += count
                    else:
                        combs[comb] = count


            # update the subwords and their frequencies
            combs = {k: combs[k] for k in combs if combs[k] > min_pair_freq}

            for comb in combs:
                for i, sw in enumerate(subwords):
                    if comb in sw:
                        new_comb = comb.replace('_', '')
                        subwords[i] = sw.replace(comb, new_comb)


        # token count
        subwords = [x.replace('_',' ##') for x in subwords]
        subword_count = {w:word_freq[w.replace(' ##','')] for w in subwords}
        unq_count = {}

        for word in subword_count:
            count = subword_count[word]
            word = word.split()

            for w in word:
                if w in unq_count:
                    unq_count[w] += count
                else:
                    unq_count[w] = count

        # get top k tokens

        self.vocab = dict(sorted(unq_count.items(), key=lambda x: x[1], reverse=True)[:self.num_tokens])
        unq_tokens = set(list(self.vocab))

        # if token's possible subwords are within the vocab but the token
        # itself isn't it's updated in the subword dictionary with its subwords

        self.subwords = {w.replace(' ##',''):w for w in subwords
                         if set.issubset(set(w.split()),unq_tokens)}

        # create a list of prefix and suffix

        self.prefix = [x for x in self.vocab if '##' not in x]
        self.suffix = [x for x in self.vocab if '##' in x]

        self.prefix = sorted(self.prefix, key=lambda x: len(x), reverse=True)
        self.suffix = sorted(self.suffix, key=lambda x: len(x), reverse=True)

        # create index mapping of tokens

        for token in self.tokens:
            if token in string.punctuation:
                self.vocab[token] = 1

        self.vocab = dict(sorted(self.vocab.items(), key=lambda x: x[1], reverse=True))

        if len(special_tokens) > 0:

            for i,tok in enumerate(special_tokens):
                self.w2i[tok] = i + len(self.special_toks)
                self.i2w[i + len(self.special_toks)] = tok

            self.special_toks.extend(special_tokens)

        for i, t in enumerate(self.vocab):
            self.w2i[t] = i + len(self.special_toks)
            self.i2w[i + len(self.special_toks)] = t


    def _split_oov(self, word):

        """
        function to split OOV token into subwords from tokenizer's vocab
        """

        if word in self.subwords:

            if self.subwords[word].replace(' ##','') == word:
                return self.subwords[word]

            else:
                return self.subwords[word]

        elif word in self.w2i:
            return word

        subwords = {}

        pref = []

        # get all possible prefix and all their corresponding suffix
        for p in self.prefix:

            if word.startswith(p):
                pref.append(p)

                subwords[p] = []

                rem = word[len(p):]
                for s in self.suffix:
                    if s[2:] in rem:
                        subwords[p].append(s[2:])

        subw = []

        for p in subwords:
            rem = re.sub(p, '', word, 1)
            suff = subwords[p]
            suff = sorted(suff, key=lambda x: len(x))[::-1]
            sorter = [(sw, rem.index(sw)) for sw in suff]
            suff = sorted(sorter, key=lambda x: x[1])
            tmp = [p]
            rpl = ''
            for s,i in suff:

                rem_word = rem.replace('_', '')

                if rem == '':
                    break

                elif s == rem[i:i+len(s)]:
                    rem = rem[:i] + '_' * len(s) + rem[i + len(s):]
                    tmp.append(s)

                elif s == rem_word[:len(s)]:

                    rpl += '_' * len(s)
                    rem = rpl + rem_word[len(s):]
                    tmp.append(s)

            fin_rem = rem.replace('_', '')

            if fin_rem in rem and "##" + fin_rem in self.vocab:
                tmp.append(fin_rem)

            subw.append(tmp)

        for i,seq in enumerate(subw):

            suffs = sorted(seq[1:],key=lambda x: len(x))[::-1]
            subw[i] = (seq[0],suffs)

        for i, (p,seq) in enumerate(subw):

            rem = word[len(p):]

            for w in seq:
                count = rem.count(w)
                rem = rem.replace(w,'_'*len(w))

                if count > 1:

                    add = [w] * (count - 1)
                    seq += add

            w_idx = []
            rem = word[len(p):]

            for w in seq:

                if w in rem:
                    idx = rem.index(w)
                    rem = rem[:idx] + '_' * len(w) + rem[idx + len(w):]
                    w_idx.append((w, idx + len(p)))
            w_idx.append((p,0))
            w_idx = sorted(w_idx, key=lambda x: x[1])
            subw[i] = [x[0] for x in w_idx]

        subw = [(''.join(x), ' ##'.join(x)) for x in subw if word[:len(''.join(x))] == ''.join(x)]

        if len(subw) == 0:
            self.subwords[word] = '<unk>'
            return '<unk>'

        elif len(subw) > 0:
            complete = [x[1] for x in subw if x[0] == word]
            if len(complete) > 0:
                fin_subw = sorted(complete,key=lambda x: len(x))[0]
                self.subwords[word] = fin_subw
                return fin_subw
            else:
                diff = [(len(word.replace(x[0],'')),x[1]) for x in subw]
                fin_subw = sorted(diff,key=lambda x: x[0])[0][1]
                self.subwords[word] = fin_subw + " <unk>"
                return fin_subw + " <unk>"


    def tokenize(self, seq):

        """
        Function for tokenizing the sequences.
        The sequence is fisrt truncated till it maxlen and then
        "<cls>" and "<sep>" tokens are added at beginning and end

        example:

        input -> "i am playing"\n
        tokenized output -> [1,5,6,7,8,2]

        :param seq: string sequence
        :return: tokenized sequence in list form
        """
        seq = np.asarray(seq.split()[:self.maxlen - 2])
        split_tok = np.vectorize(self._split_oov)
        seq = split_tok(seq)
        seq = ' '.join(seq).split()

        if len(seq) > self.maxlen - 2:
            seq = seq[:self.maxlen]

        seq = np.asarray(['<cls>'] + ' '.join(seq).split() + ['<sep>']).astype('object')
        seq = [self.w2i[t] for t in seq]

        return seq

    def add_padding(self, seq):
        """
        pad sequence to maxlen
        :param seq: python list as tokenized sequence
        :return: padded tokenized sequence as 1d array
        """
        if len(seq) > self.maxlen:
            seq = seq[:self.maxlen]
            seq[-1] = self.w2i['<sep>']
            return seq
        elif len(seq) < self.maxlen:
            seq = seq + [0 for _ in range(self.maxlen - len(seq))]
            return seq
        else:
            return seq

    def load_tokenizer(self,file_path):
        """
        load attributes from pkl file
        :param file_path: path to json file
        """
        with open(file_path,'rb') as f:
            data = pickle.load(f)
        self.__dict__.update(data)

    def save_tokenizer(self,file_path):
        """
        save attributes to pkl file
        :param file_path: path to json file
        """
        attributes = self.__dict__
        with open(file_path,'wb') as f:
            pickle.dump(attributes,f)


class BytePairEncodingTokenizer:
    """
    Byte pair encoding tokenization

    The tokenizer would be first called and then trained
    to learn the subwords based on the sequences. Once done,
    the tokenizer will first split word and then tokenize it.

    In case of any OOV word, the tokenizer will try break it
    into known subwords

    example: playing -> play ing

    """
    def __init__(self,num_tokens=None,maxlen=None,max_word_count=None):

        """

        :param num_tokens: no. of tokens within the vocabulary
        :param maxlen: maximum sequence length during tokenization
        :param max_word_count: max number of tokens for tokenizer training
        """

        self.special_toks = ['<pad>', '<cls>', '<sep>', '<unk>', '<mask>']
        self.w2i = {x: i for i, x in enumerate(self.special_toks)}
        self.i2w = {self.w2i[k]: k for k in self.w2i}
        self.size = len(self.special_toks)
        self.tokens = {}
        self.words = {}
        self.vocab = {}
        self.subwords = {}
        self.num_tokens = num_tokens
        self.maxlen = maxlen
        self.max_word_count = max_word_count
        if max_word_count is not None:
            assert max_word_count > num_tokens

    def __call__(self, text):
        """
        call function for gathering the unique words

        :param text: list of sentences
        """

        # get corpus

        corpus = ' '.join(text).split()
        tok_freq = Counter(corpus)
        tokens = sorted(corpus, key=lambda x: (tok_freq[x], x), reverse=True)

        count = 1

        # obtain word counts

        for i in range(1, len(tokens)):

            if tokens[i] == tokens[i - 1]:
                count += 1
            else:
                self.tokens[tokens[i - 1]] = count

                if tokens[i - 1].isalnum():
                    self.words[tokens[i - 1]] = count

                count = 1

            if len(self.words) == self.num_tokens and self.max_word_count is None:
                break

            elif self.max_word_count is not None and len(self.words) == self.max_word_count:
                break

    def train(self, iterations=2, min_pair_freq=200,
              strategy='coverage',special_tokens=[]):

        """
        Tokenizer training function
        The tokenizer learns the subword sequences and keeps the once as per
        their frequency threshold

        :param iterations: number of training iterations
        :param min_pair_freq: minimum frequency for subword pairs
        :param strategy: strategy for creating subwords. Frequency-based will create subwords
         based on frequency of the corpus and coverage-based will create based on number unique
         tokens subword is found in.
        :param special_tokens: list of new special tokens to be added
        """

        if strategy == 'coverage':
            word_freq = {k:1 for k in self.words}

        elif strategy == 'frequency':
            word_freq = self.words.copy()

        else:
            print(f'error : `{strategy}` is not valid')
            return

        if len(special_tokens) > 0:
            for tok in special_tokens:
                if tok in word_freq:
                    del word_freq[tok]

        # get subword pairs

        subwords = [w.replace('', '_')[1:-1] for w in word_freq]

        # iterate, create new subwords and update

        for _ in range(iterations):

            combs = {} # for storing split combination

            for i, w in enumerate(subwords):

                w = w.split('_')
                for j, _ in enumerate(w):
                    count = word_freq[subwords[i].replace('_', '')]
                    comb = '_'.join(w[j:j + 2])
                    if comb in combs:
                        combs[comb] += count
                    else:
                        combs[comb] = count

            combs = {k: combs[k] for k in combs if combs[k] > min_pair_freq}

            for comb in combs:
                for i, sw in enumerate(subwords):
                    if comb in sw:
                        new_comb = comb.replace('_', '')
                        subwords[i] = sw.replace(comb, new_comb)

        # update vocabulary

        for i, sw in enumerate(subwords):
            sw = sw.split('_')
            count = word_freq[''.join(sw)]
            for x in sw:
                if x not in self.vocab:
                    self.vocab[x] = count
                else:
                    self.vocab[x] += count

        self.vocab = dict(sorted(self.vocab.items(), key=lambda x: x[1], reverse=True))
        top_k_words = set(list(self.vocab)[:self.num_tokens])

        self.vocab = {k:v for k,v in self.vocab.items() if k in top_k_words}

        for token,count in self.tokens.items():
            if token in string.punctuation:
                self.vocab[token] = count

        self.vocab = dict(sorted(self.vocab.items(), key=lambda x: x[1], reverse=True))

        # update subword splits

        for subw in subwords:

            word = subw.replace('_','')
            subw = subw.split('_')

            if set.issubset(set(subw),top_k_words):
                self.subwords[word] = ' '.join(subw)

        # add new special tokens

        if len(special_tokens) > 0:
            for tok in special_tokens:
                self.special_toks.append(tok)
                self.i2w[len(self.special_toks)-1] = tok
                self.w2i[tok] = len(self.special_toks)-1

        # update word indices

        for i,word in enumerate(self.vocab):

            i += len(self.special_toks)

            self.i2w[i] = word
            self.w2i[word] = i

    def _split_oov(self, word):

        """
        function to split an OOV token

        :param word: python string
        :return: subword split
        """

        if word in self.w2i:
            return word

        elif word in self.subwords:
            return self.subwords[word]

        elif word in string.punctuation:
            return '<unk>'

        subwords = {} # for storing all possible subwords

        # get all possible subword sequences

        for p in self.vocab:
            tmp = []
            if (p in word) and (p not in subwords):
                if word.index(p) == 0:
                    subwords[p] = []
                    sw = re.sub(p, '', word, 1)
                    for s in self.vocab:
                        if (s in sw) and (s not in subwords[p]):
                            subwords[p].append(s)
                else:
                    pass

        subw = [] # storing different subword sequence

        # keep only required subwords

        for p in subwords:

            rem = re.sub(p, '', word, 1)
            suff = subwords[p]
            suff = sorted(suff, key=lambda x: len(x))[::-1]
            sorter = [(sw, rem.index(sw)) for sw in suff]
            suff = sorted(sorter, key=lambda x: x[1])

            suff_order = suff
            tmp = [p]
            rpl = ''

            for x in suff_order:

                s, i = x
                rem_word = rem.replace('_', '')

                if rem == '':
                    break

                elif s == rem_word[:len(s)]:
                    rpl += '_' * len(s)
                    rem = rpl + rem_word[len(s):]
                    tmp.append(s)

                elif s == rem[i:i+len(s)]:
                    rem = rem[:i] + '_' * len(s) + rem[i+len(s):]
                    tmp.append(s)

            fin_rem = rem.replace('_', '')

            if fin_rem in rem and fin_rem in self.vocab:
                tmp.append(fin_rem)

            subw.append(tmp)

        # sort the subwords as per the sequence of the word

        subw = [sorted(list(set(seq)),key=lambda x: len(x))[::-1] for seq in subw]

        for i,seq in enumerate(subw):

            rem = word

            for w in seq:
                count = rem.count(w)
                rem = rem.replace(w,'_'*len(w))

                if count > 1:

                    add = [w] * (count - 1)
                    seq += add

            rem = word

            w_idx = []

            for w in seq:

                if w in rem:

                    idx = rem.index(w)
                    rem = rem[:idx] + '_' * len(w) + rem[idx + len(w):]
                    w_idx.append((w,idx))

            w_idx = sorted(w_idx, key=lambda x: x[1])

            subw[i] = [x[0] for x in w_idx]

        subw = [(''.join(x),' '.join(x)) for x in subw if word[:len(''.join(x))] == ''.join(x)]

        # return output as per the subword sequence obtained

        if len(subw) == 0:
            self.subwords[word] = '<unk>'
            return '<unk>'

        elif len(subw) > 0:
            complete = [x[1] for x in subw if x[0] == word]
            if len(complete) > 0:
                fin_subw = sorted(complete,key=lambda x: len(x))[0]
                self.subwords[word] = fin_subw
                return fin_subw
            else:
                diff = [(len(word.replace(x[0],'')),x[1]) for x in subw]
                fin_subw = sorted(diff,key=lambda x: x[0])[0][1]
                self.subwords[word] = fin_subw + " <unk>"
                return fin_subw + " <unk>"

    def tokenize(self, seq):
        """
        Function for tokenizing the sequences.
        The sequence is fisrt truncated till it maxlen and then
        "<cls>" and "<sep>" tokens are added at beginning and end

        example:

        input -> "i am playing"
        output before tokenization -> "<cls> i am play ing <sep>"
        tokenized output -> [1,5,6,7,8,2]

        :param seq: string sequences in a list
        :return: tokenized sequences in a list
        """
        seq = np.asarray(seq.split()[:self.maxlen-2])
        split_tok = np.vectorize(self._split_oov)
        seq = split_tok(seq)
        seq = seq[:self.maxlen - 2]
        seq = ['<cls>'] + ' '.join(seq).split() + ['<sep>']
        seq = [self.w2i[s] for s in seq]
        return seq

    def add_padding(self,seq):
        """
        Function for adding padding to the sequence
        :param seq: python list as tokenized sequence
        :return: padded tokenized sequence as 1d array
        """

        if len(seq) > self.maxlen:
            seq = seq[:self.maxlen]
            seq[-1] = self.w2i['<sep>']
            return seq
        elif len(seq) < self.maxlen:
            seq = seq + [0 for _ in range(self.maxlen - len(seq))]
            return seq
        else:
            return seq

    def load_tokenizer(self,file_path):
        """
        load attributes from pkl file
        :param file_path: path to json file
        """
        with open(file_path,'rb') as f:
            data = pickle.load(f)
        self.__dict__.update(data)

    def save_tokenizer(self,file_path):
        """
        save attributes to pkl file
        :param file_path: path to json file
        """
        attributes = self.__dict__
        with open(file_path,'wb') as f:
            pickle.dump(attributes,f)




