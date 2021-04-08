import re
import itertools
import numpy as np

from abc import abstractmethod

from skimage.filters import sobel
from skimage.color import rgb2gray, gray2rgb
from skimage.segmentation import quickshift, watershed


class TextNeighborhoodGenerator(object):

    def __init__(self, bb_predict, ocr=0.1):
        self.bb_predict = bb_predict
        self.ocr = ocr  # other class ratio

    @abstractmethod
    def generate(self, text, num_samples=1000):
        return


class IndexedText(object):
    """String with various indexes."""

    def __init__(self, text, split_expression=r'\W+', bow=True):
        """Initializer.

        Args:
            text: string with text text in it
            split_expression: Regex string or callable. If regex string, will be used with re.split.
                If callable, the function should return a list of tokens.
            bow: if True, a word is the same everywhere in the text - i.e. we
                 will index multiple occurrences of the same word. If False,
                 order matters, so that the same word will have different ids
                 according to position.
        """

        if callable(split_expression):
            tokens = split_expression(text)
            self.as_list = self._segment_with_tokens(text, tokens)
            tokens = set(tokens)

            def non_word(string):
                return string not in tokens

        else:
            # with the split_expression as a non-capturing group (?:), we don't need to filter out
            # the separator character from the split results.
            self.as_list = re.split(r'(%s)|$' % split_expression, text)
            non_word = re.compile(r'(%s)|$' % split_expression).match

        self.as_np = np.array(self.as_list)
        self.string_start = np.hstack(
            ([0], np.cumsum([len(x) for x in self.as_np[:-1]])))
        vocab = {}
        self.inverse_vocab = []
        self.positions = []
        self.bow = bow
        non_vocab = set()
        for i, word in enumerate(self.as_np):
            if word in non_vocab:
                continue
            if non_word(word):
                non_vocab.add(word)
                continue
            if bow:
                if word not in vocab:
                    vocab[word] = len(vocab)
                    self.inverse_vocab.append(word)
                    self.positions.append([])
                idx_word = vocab[word]
                self.positions[idx_word].append(i)
            else:
                self.inverse_vocab.append(word)
                self.positions.append(i)
        if not bow:
            self.positions = np.array(self.positions)

    # def raw_string(self):
    #     """Returns the original text string"""
    #     return self.text

    def num_words(self):
        """Returns the number of tokens in the vocabulary for this document."""
        return len(self.inverse_vocab)

    def word(self, id_):
        """Returns the word that corresponds to id_ (int)"""
        return self.inverse_vocab[id_]

    def string_position(self, id_):
        """Returns a np array with indices to id_ (int) occurrences"""
        if self.bow:
            return self.string_start[self.positions[id_]]
        else:
            return self.string_start[[self.positions[id_]]]

    def inverse_removing(self, words_to_remove):
        """Returns a string after removing the appropriate words.

        If self.bow is false, replaces word with UNKWORDZ instead of removing
        it.

        Args:
            words_to_remove: list of ids (ints) to remove

        Returns:
            original text string with appropriate words removed.
        """
        mask = np.ones(self.as_np.shape[0], dtype='bool')
        mask[self.__get_idxs(words_to_remove)] = False
        if not self.bow:
            return ''.join([self.as_list[i] if mask[i] else 'UNKWORDZ' for i in range(mask.shape[0])])
        return ''.join([self.as_list[v] for v in mask.nonzero()[0]])

    @staticmethod
    def _segment_with_tokens(text, tokens):
        """Segment a string around the tokens created by a passed-in tokenizer"""
        list_form = []
        text_ptr = 0
        for token in tokens:
            inter_token_string = []
            while not text[text_ptr:].startswith(token):
                inter_token_string.append(text[text_ptr])
                text_ptr += 1
                if text_ptr >= len(text):
                    raise ValueError("Tokenization produced tokens that do not belong in string!")
            text_ptr += len(token)
            if inter_token_string:
                list_form.append(''.join(inter_token_string))
            list_form.append(token)
        if text_ptr < len(text):
            list_form.append(text[text_ptr:])
        return list_form

    def __get_idxs(self, words):
        """Returns indexes to appropriate words."""
        if self.bow:
            return list(itertools.chain.from_iterable(
                [self.positions[z] for z in words]))
        else:
            return self.positions[words]


class TextLimeGenerator(TextNeighborhoodGenerator):

    def __init__(self, bb_predict, ocr=0.1, bow=True, split_expression=r'\W+'):
        super(TextLimeGenerator, self).__init__(bb_predict, ocr)
        self.bow = bow
        self.split_expression = split_expression

    def generate(self, text, num_samples=1000, hide_color=None):
        indexed_text = IndexedText(text, bow=self.bow, split_expression=self.split_expression)
        nbr_features = indexed_text.num_words()
        feature_names = range(nbr_features)

        nbr_words_to_suppres_list = np.random.randint(1, nbr_features, num_samples - 1)
        Z = np.ones((num_samples, nbr_features))
        Z[0] = np.ones(nbr_features)
        Z_text = [text]
        for i, nbr_words_to_suppres in enumerate(nbr_words_to_suppres_list, start=1):
            words_to_suppress_indexes = np.random.choice(feature_names, nbr_words_to_suppres, replace=False)
            Z[i, words_to_suppress_indexes] = 0
            Z_text.append(indexed_text.inverse_removing(words_to_suppress_indexes))

        Yb = self.bb_predict(Z_text)
        class_value = Yb[0]

        Z, Z_text = self.__balance_neigh(indexed_text, Z, Z_text, Yb, num_samples, class_value, nbr_features)
        Yb = self.bb_predict(Z_text)

        return Z, Yb, class_value, indexed_text

    def __balance_neigh(self, indexed_text, Z, Z_text, Yb, num_samples, class_value, nbr_features):
        class_counts = np.unique(Yb, return_counts=True)
        if len(class_counts[0]) <= 2:
            ocs = int(np.round(num_samples * self.ocr))
            Z1, Z1_text = self.__rndgen_not_class(indexed_text, ocs, class_value, nbr_features)
            if len(Z1) > 0:
                Z = np.concatenate((Z, Z1), axis=0)
                Z_text.extend(Z1_text)
        else:
            max_cc = np.max(class_counts[1])
            max_cc2 = np.max([cc for cc in class_counts[1] if cc != max_cc])
            if max_cc2 / len(Yb) < self.ocr:
                ocs = int(np.round(num_samples * self.ocr)) - max_cc2
                Z1, Z1_text = self.__rndgen_not_class(indexed_text, ocs, class_value, nbr_features)
                if len(Z1) > 0:
                    Z = np.concatenate((Z, Z1), axis=0)
                    Z_text.extend(Z1_text)
        return Z, Z_text

    def __rndgen_not_class(self, indexed_text, num_samples, class_value, nbr_features, max_iter=1000):
        Z = list()
        Z_text = list()
        iter_count = 0
        feature_names = range(nbr_features)
        while len(Z) < num_samples:
            z = np.ones(nbr_features)
            nbr_words_to_suppres = np.random.randint(1, nbr_features)
            words_to_suppress_indexes = np.random.choice(feature_names, nbr_words_to_suppres, replace=False)
            z[words_to_suppress_indexes] = 0
            z_text = indexed_text.inverse_removing(words_to_suppress_indexes)
            if self.bb_predict([z_text])[0] != class_value:
                Z.append(z)
                Z_text.append(z_text)
            iter_count += 1
            if iter_count >= max_iter:
                break

        Z = np.array(Z)
        return Z, Z_text
