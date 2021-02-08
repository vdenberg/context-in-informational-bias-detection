from __future__ import absolute_import, division, print_function
import os
import sys
import logging
import csv
from lib.handle_data.BasilLoader import load_basil_spans
from lib.utils import standardise_id
import re

logger = logging.getLogger()
csv.field_size_limit(2147483647) # Increase CSV reader's field limit incase we have long text.

PAD_TOKEN = '<pad>'


def convert_basil_for_plm_inputs(basil, task='sent_clf', ofp='data/tok_clf/plm_basil.tsv'):
    """
    Select relevant columns for input to huggingface implementations of Pre-trained Language Models
    BERT and RoBERTa for Sentence and Token classification.
    :param
    basil: original BASIL DataFrame
    task: token or sentence classification
    ofp: output file path of all instances
    :return: None, writes to ofp
    """
    basil['id'] = basil['uniq_idx.1'].str.lower()
    basil['alpha'] = ['a'] * len(basil)

    if task == 'sent_clf' or task == 'seq_sent_clf':
        basil = basil.rename(columns={'bias': 'label'})
    elif task == 'tok_clf':
        basil = basil.rename(columns={'inf_start_ends': 'label'})

    basil[['id', 'label', 'alpha', 'sentence']].to_csv(ofp, sep='\t', index=False, header=False)


class SpanToBio():
    """ Gives BIO tags corresponding to spans """

    def __init__(self, tok):
        self.spacy_tokenizer = tok  # spacy.load("en_core_web_sm")

    def tokenize(self, sent):
        if not isinstance(sent, float):
            sent = re.sub('  ', ' ', sent)
            return [token.text for token in self.spacy_tokenizer(sent)]
        else:
            return []

    def get_char_mapping(self, sent, toks):
        """ Returns a list of len(sentence) but instead of characters
        it has the number of the token that character belongs to
        E.g: "The man sleeps", "000 111 222"""

        tok_in_progress = ''
        current_tok_nr = 0
        mapping = []

        for char in sent:

            if char == ' ':
                mapping_value = ' '

            else:
                tok_in_progress += char
                mapping_value = current_tok_nr
                if tok_in_progress == toks[0]:
                    # token finished, update counter
                    current_tok_nr += 1
                    # remove from list of tokens that still need to be matches
                    toks = toks[1:]
                    # clear temp token
                    tok_in_progress = ''

            mapping.append(mapping_value)

        return mapping

    def get_lab_seq(self, sent, toks, spans):
        token_indices = range(len(toks))
        lab_database = {i: 0 for i in token_indices}
        lab_seq = []

        for span in spans:
            # apply span to sequence of token ids & reduce sequence of ids to set of ids that fall within span
            classified_chars = self.get_char_mapping(sent, toks)
            cl_chars_in_span = classified_chars[span[0]:span[1]]
            tok_idxs_in_span = set(cl_chars_in_span) - set(' ')

            # mark for each token whether it is a member of that set
            current_seq = [int(i in tok_idxs_in_span) for i in token_indices]
            for i in token_indices:
                if bool(current_seq[i]):
                    lab_database[i] = 1

        lab_seq = [lab_database[i] for i in token_indices]

        return lab_seq

    def lab_seq_to_bio_tags(self, lab_seq):
        # convert to bio tags
        bio_tags = []
        prev_bio_tag = None

        for lab in lab_seq:

            bio_tag = 'O'

            if lab == 1:
                if prev_bio_tag in [None, 'O']:
                    bio_tag = 'B-BIAS'

                if prev_bio_tag in ['B-BIAS', 'I-BIAS']:
                    bio_tag = 'I-BIAS'

            bio_tags.append(bio_tag)
            prev_bio_tag = bio_tag

        return bio_tags

    def span_to_bio(self, sent, spans):
        toks = self.tokenize(sent)
        spans = load_basil_spans(spans)

        if spans:
            lab_seq = self.get_lab_seq(sent, toks, spans)
        else:
            lab_seq = ['O' for t in toks]
        bio_tags = self.lab_seq_to_bio_tags(lab_seq)
        return toks, bio_tags


def convert_to_bio(sentences, spans):
    """ Converts span annotations to BIO tags
    For example:
    #df = pd.read_csv('data/basil.csv', index_col=0)

    sentences = df.sentence
    spans = df.inf_start_ends
    bio_tags = convert_to_bio(sentences, spans)
    """
    sph = SpanToBio()

    all_bio_tags = []
    for sent, spans in zip(sentences, spans):
        toks, bio_tags = sph.span_to_bio(sent, spans)

        bio_tags_as_string = " ".join(bio_tags)
        all_bio_tags.append(bio_tags_as_string)

    return all_bio_tags


def expand_to_wordpieces(original_sentence, original_labels, tokenizer):
    """
    Maps a BIO Label Sequence to the BERT WordPieces length preserving the BIO Format
    :param original_sentence: String of complete-word tokens separated by spaces
    :param original_labels: List of labels 1-1 mapped to tokens
    :param tokenizer: BertTokenizer with do_basic_tokenize=False to respect the original_sentence tokenization.
    :return:
    """

    word_pieces = tokenizer.tokenize(original_sentence)

    tmp_labels, lbl_ix = [], 0
    for tok in word_pieces:
        # if "##" in tok:
        if lbl_ix != 0 and not tok.startswith('Ġ'):
            tmp_labels.append("X")
        else:
            try:
                tmp_labels.append(original_labels[lbl_ix])
            except IndexError:
                print('Index Error at index', lbl_ix)
                print('Original sentence:', original_sentence)
                print('Original label split:', len(original_sentence.split(' ')), original_sentence.split(' '))
                print('Original label:', len(original_labels), original_labels)
                print('Word pieces:', len(word_pieces), word_pieces)
                print('tmp_labels:', len(tmp_labels), tmp_labels)
                exit(0)
            lbl_ix += 1

    expanded_labels = []
    for i, lbl in enumerate(tmp_labels):
        if lbl == "X":
            # prev = tmp_labels[i-1]
            prev = expanded_labels[-1]
            if prev.startswith("B-"):
                expanded_labels.append("I-" + prev[2:])
            else:
                expanded_labels.append(prev)
        else:
            expanded_labels.append(lbl)

    assert len(word_pieces) == len(expanded_labels)

    #('Expanded labels:', expanded_labels)
    #if original_sentence.startswith('Huckabee , an ordained'):
        #exit(0)

    return word_pieces, expanded_labels


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, my_id, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.my_id = my_id
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, sep='\t', quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=sep, quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(cell for cell in line)
                if input_file.endswith('basil.csv'):
                    line = line[1:]
                lines.append(line)
            return lines


class BinaryClassificationProcessor(DataProcessor):
    """Processor for binary classification dataset."""

    def get_examples(self, fp, name, sep):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(fp), sep), name)

    def get_labels(self, output_mode):
        """See base class."""
        if output_mode == 'tok_clf':
            return ["O", "B-BIAS", "I-BIAS"]
        else:
            return ["0", "1"]

    def _create_examples(self, lines, set_type, dataset='basil'):
        """Creates examples for the training and dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            my_id = standardise_id(line[0])
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, my_id=my_id, text_a=text_a, text_b=None, label=label))
        return examples


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, my_id, input_ids, input_mask, segment_ids, label_id):
        self.my_id = my_id
        if my_id and my_id != 1:
            my_id = standardise_id(my_id)
            self.sent_id = int(my_id[-2:])
            self.article = my_id[:-2]
        else:
            self.sent_id = None
            self.article = None
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_example_to_roberta_feature(example_row):
    # return example_row
    example, label_map, max_seq_length, tokenizer, spacy_tokenizer, output_mode = example_row

    # tokens
    if output_mode == 'tok_clf':
        sp2bio = SpanToBio(spacy_tokenizer)
        spacy_tokens, spacy_labels = sp2bio.span_to_bio(example.text_a, example.label)
        assert len(spacy_tokens) == len(spacy_labels)

        tokens_a = " ".join(spacy_tokens)
        labels = spacy_labels
        tokens_a, labels = expand_to_wordpieces(tokens_a, labels, tokenizer)

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
    else:
        tokens_a = example.text_a

    encoded = tokenizer.encode_plus(tokens_a, max_length=max_seq_length, pad_to_max_length=True,
                                    add_special_tokens=True) #truncation=True,

    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    assert len(input_ids) == max_seq_length
    assert len(attention_mask) == max_seq_length

    # labels
    if output_mode == "sent_clf" or output_mode == "seq_sent_clf":
        label_id = label_map[example.label]

    elif output_mode == 'tok_clf':
        labels = ['O'] + labels + ['O']
        label_id = [label_map.get(lab) for lab in labels]
        padding = [0] * (max_seq_length - len(label_id))
        label_id += padding  # cls=0, pad=1

        assert len(label_id) == max_seq_length

    else:
        raise KeyError(output_mode)

    return InputFeatures(my_id=example.my_id,
                         input_ids=input_ids,
                         input_mask=attention_mask,
                         segment_ids=[],
                         label_id=label_id)


def convert_example_to_bert_feature(example_row):
    # return example_row
    example, label_map, max_seq_length, tokenizer, spacy_tokenizer, output_mode = example_row

    # tokens

    if output_mode == 'tok_clf':
        sp2bio = SpanToBio(spacy_tokenizer)
        spacy_tokens, spacy_labels = sp2bio.span_to_bio(example.text_a, example.label)
        assert len(spacy_tokens) == len(spacy_labels)

        tokens_a = " ".join(spacy_tokens)
        labels = spacy_labels
        tokens_a, labels = expand_to_wordpieces(tokens_a, labels, tokenizer)

    else:
        tokens_a = tokenizer.tokenize(example.text_a)

    #tokens_b = None
    #if example.text_b:
        #    tokens_b = tokenizer.tokenize(example.text_b)
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
    #   _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    #else:

    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[:(max_seq_length - 2)]    # Account for [CLS] and [SEP] with "- 2"

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]

    # segment ids

    segment_ids = [0] * len(tokens)

    #if tokens_b:
    #    tokens += tokens_b + ["[SEP]"]
    #    segment_ids += [1] * (len(tokens_b) + 1)

    # input ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)  # The mask has 1 for real tokens and 0 for padding tokens.

    # padding

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    # labels
    if output_mode == "sent_clf":
        label_id = label_map[example.label]

    elif output_mode == "tok_clf":
        labels = ['O'] + labels + ['O']

        label_id = [label_map.get(lab) for lab in labels]
        label_id += padding

        assert len(label_id) == max_seq_length

    else:
        raise KeyError(output_mode)

    return InputFeatures(my_id=example.my_id,
                         input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         label_id=label_id)