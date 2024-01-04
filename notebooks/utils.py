import os
import re
import shlex
import subprocess
from collections import Counter
from functools import reduce

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from nltk.corpus import words

import docx
import spacy


nlp = spacy.load('en_core_web_lg')

spacy.tokens.Token.set_extension('feature', default=None)

WORD_SET = set(words.words())

def read_data(PATH, print_exception=False):
    data = pd.DataFrame(columns=['pidn', 'variant', 'text']).set_index('pidn')

    for variant in os.listdir(PATH):
        for fname in os.listdir(os.path.join(PATH, variant)):
            try:
                pidn = int(fname[:-4])
                with open(os.path.join(os.path.join(PATH, variant), fname), 'r') as f:
                    text = f.read()
                data.loc[(pidn, 0), :] = variant, text
            except Exception as e:
                if print_exception:
                    print(pidn, e)

    data = data.sort_index()

    return data

def read_docx(PATH, affix='', print_exception=False):
    data = pd.DataFrame(columns=['pidn', 'visit', 'variant', 'raw text']).set_index(['pidn', 'visit'])

    for variant in os.listdir(PATH):
        if variant == '.DS_Store':
            continue
        for fname in os.listdir(os.path.join(PATH, variant + affix)):
            try:
                pidn = int(fname[:5])
                if variant == 'long_nfvPPA':
                    visit = 1
                else:
                    visit = 0
                doc = docx.Document(os.path.join(os.path.join(PATH, variant + affix), fname))
                text = '\n'.join([x.text for x in doc.paragraphs if x.text != ''])
                data.loc[(pidn ,visit), :] = variant, text
            except Exception as e:
                if print_exception:
                    print(pidn, e)

    data = data.sort_index()

    return data

def get_ft_label(text):

    return subprocess.check_output("""~/fastText/fasttext predict-prob ~/fastText/amazon_review_polarity.bin - <<< {}""".format(shlex.quote(text)), shell=True).decode('utf8')[:-1]

# def aggregate(data):

#     data.loc[:, 'sentiment'] = pd.Series(get_ft_label('\n'.join(data.loc[:, 'text'].apply(lambda x: x.replace('\n', '')))).split('\n'), index=data.index).str[9].astype(int) - 1

#     parts_of_speech = reduce(lambda x, y: x.append(y, sort=True), data.loc[:, 'doc'].apply(lambda x: pd.DataFrame({k: [v] for k, v in dict(Counter([token.pos_ for token in x])).items()})))
#     parts_of_speech.index = data.index
#     parts_of_speech = parts_of_speech.fillna(0)
#     data = data.join(parts_of_speech)

#     features = reduce(lambda x, y: x.append(y, sort=True), data.loc[:, 'doc'].apply(lambda x: pd.DataFrame({k: [v] for k, v in dict(Counter([token._.feature for token in x if token._.feature])).items()})))
#     features.index = data.index
#     features = features.fillna(0)
#     data = data.join(features)

#     return data

matcher1 = spacy.matcher.Matcher(nlp.vocab, validate=True)
matcher1.add('there be', [[{'LOWER': 'there'}, {'POS': 'AUX'}]])
matcher1.add('i see', [[{'LOWER': 'i'}, {'POS': 'VERB', 'LEMMA': 'see'}]])
matcher1.add('not argument', [[{'LOWER': 'i'}, {'POS': 'VERB', 'LEMMA': 'see'}, {'DEP': {'IN': ['det', 'amod', 'advmod']}, 'OP': '*'}, {'DEP': 'dobj'}]])
matcher1.add('appear to be', [[{'LEMMA': 'appear'}, {'LEMMA': 'to'}, {'LEMMA': 'be'}]])

matcher2 = spacy.matcher.Matcher(nlp.vocab, validate=True)
matcher2.add('aux', [[{'POS': 'AUX', 'DEP': {'IN': ['aux', 'auxpass']}}]])
matcher2.add('copula', [[{'POS': 'AUX', 'DEP': {'NOT_IN': ['aux', 'auxpass']}}, {'POS': {'IN': ['ADV', 'PART']}, 'OP': '*'}, {'POS': {'NOT_IN': ['ADV', 'PART']}, 'DEP': {'NOT_IN': ['cc']}}]])
matcher2.add('verb', [[{'POS': 'VERB'}]])
matcher2.add('argument', [[{'DEP': 'dobj'}], [{'LOWER': 'i'}, {'POS': 'VERB', 'LEMMA': 'see'}, {'DEP': {'IN': ['det', 'amod', 'advmod']}, 'OP': '*'}, {'DEP': 'nsubj'}]])
matcher2.add('predicate', [[{'POS': 'AUX'}, {'DEP': {'IN': ['det', 'amod', 'advmod']}, 'OP': '*'}, {'DEP': 'attr'}]])
matcher2.add('adjunct', [[{'DEP': 'pobj'}]])

# def add_feature(doc):

#     for x in matcher1(doc):
#         if nlp.vocab.strings[x[0]] in ('there be', 'i see', 'not argument'):
#             doc[x[1]:x[2]][-1]._.feature = nlp.vocab.strings[x[0]]
#         elif not doc[x[1]:x[2]][0]._.feature:
#             doc[x[1]:x[2]][0]._.feature = nlp.vocab.strings[x[0]]

#     for x in matcher2(doc):
#         if nlp.vocab.strings[x[0]] in ('predicate', 'argument') and not doc[x[1]:x[2]][-1]._.feature:
#             doc[x[1]:x[2]][-1]._.feature = nlp.vocab.strings[x[0]]
#         elif not doc[x[1]:x[2]][0]._.feature:
#             doc[x[1]:x[2]][0]._.feature = nlp.vocab.strings[x[0]]

def add_feature(doc):

    def visit(word):

        if word.pos_ == 'AUX':
            if word.dep_ == 'aux':
                word._.feature = 'aux' # Syntactic auxiliary verb
            elif word.n_lefts > 0 and next(word.lefts).lemma_ == 'there':
                word._.feature = 'there be'
            elif word.n_rights > 0 and next(word.rights).dep_ != 'cc':
                word._.feature = 'copula'
            else:
                word._.feature = 'other aux'
        elif word.pos_ == 'VERB':
            if word.lemma_ == 'see' and word.n_lefts > 0 and next(word.lefts).text.upper() == 'I':
                word._.feature = 'i see'
            elif word.lemma_ != 'see' and word.n_lefts > 0 and next(word.lefts).text.upper() == 'I':
                word._.feature = 'i ...'
            elif word.lemma_ == 'appear' and word.n_rights > 0 and next(word.rights).pos_ == 'AUX':
                word._.feature = 'appear to be'
            elif word.lemma_ == 'look' and word.n_rights > 0 and next(word.rights).pos_ == 'SCONJ':
                word._.feature = 'look like'
            elif word.dep_ == 'ccomp':
                word._.feature = 'verb'
            else:
                word._.feature = 'verb'
        elif word.pos_ == 'NOUN' or word.pos_ == 'NUM':
            try:
                anc = next(word.ancestors)
                if anc.pos_ == 'VERB' and word.dep_ == 'dobj':
                    if anc._.feature and anc._.feature == 'i see':
                        word._.feature = 'predicate'
                    else:
                        word._.feature = 'argument'
                elif anc.pos_ == 'ADP' and word.dep_ == 'pobj':
                    if anc.text.lower() != 'of':
                        try:
                            anc_anc = next(anc.ancestors)
                            if anc_anc.pos_ == 'AUX' and len([x for x in anc_anc.rights if x.pos_ == 'ADP']) == 1:
                                word._.feature = 'argument'
                            else:
                                word._.feature = 'adjunct'
                        except:
                            word._.feature = 'adjunct'
                    else:
                        try:
                            anc_anc = next(anc.ancestors)
                            if anc_anc._.feature:
                                word._.feature = anc_anc._.feature
                                anc_anc._.feature = '... of'
                        except:
                            pass
                elif anc.pos_ == 'AUX' and word.dep_ == 'attr' or word.dep_ == 'dobj':
                    word._.feature = 'predicate'
                elif anc.pos_ == 'SCONJ' and word.dep_ == 'pobj':
                    try:
                        anc_anc = next(anc.ancestors)
                        if anc_anc.pos_ == 'AUX' or anc_anc.lemma_ == 'look':
                            word._.feature = 'predicate'
                        else:
                            word._.feature = 'adjunct'
                    except:
                        pass
                elif anc._.feature and 'conj' not in anc._.feature and word.dep_ == 'conj':
                    if anc.pos_ != 'VERB':
                        word._.feature = anc._.feature + ' conj'
                    else:
                        try:
                            anc_anc = next(anc.ancestors)
                            if anc_anc._.feature and anc_anc.pos_ != 'VERB':
                                word._.feature = anc_anc._.feature + ' conj'
                        except:
                            pass
            except:
                pass
        elif word.pos_ == 'ADV' and word.dep_ == 'advmod':
            try:
                anc = next(word.ancestors)
                if anc.pos_ not in ('ADV', 'AUX'):
                    word._.feature = 'adv adjunct'
                elif anc.pos_ == 'AUX':
                    n_nouns = 0
                    n_advs = 0
                    for right in anc.rights:
                        if right.pos_ == 'NOUN':
                            n_nouns += 1
                        elif right.pos_ == 'ADV':
                            n_advs += 1
                    if n_nouns > 0:
                        word._.feature = 'adv adjunct'
                    elif n_advs == 1:
                        word._.feature = 'predicate'
            except:
                pass
        elif word.pos_ == 'ADJ' and word.dep_ == 'acomp':
            word._.feature = 'predicate'

        if word._.feature == 'adjunct':
            try:
                anc = next(word.ancestors)
                if anc.lemma_ == 'into':
                    try:
                        anc_anc = next(anc.ancestors)
                        if anc_anc.lemma_ == 'pour':
                            word._.feature = 'argument'
                    except:
                        pass
            except:
                pass

        for child in word.children:
            visit(child)


    for sent in doc.sents:
        if len(sent) > 0:
            visit(sent.root)

def get_text_with_features(doc):

    def join_punctuation(seq, characters='.,;?!'):
        characters = set(characters)
        seq = iter(seq)
        current = next(seq)

        for nxt in seq:
            if nxt in characters:
                current += nxt
            else:
                yield current
                current = nxt

        yield current

    result = ' '.join(join_punctuation([token.text + ('<{}>'.format(token._.feature) if token._.feature else '') for token in doc]))
    result = re.sub(r'\n\ +', '\n', result)
    return result

def get_text_with_pos(doc, pos='pos'):

    def join_punctuation(seq, characters='.,;?!'):
        characters = set(characters)
        seq = iter(seq)
        current = next(seq)

        for nxt in seq:
            if nxt in characters:
                current += nxt
            else:
                yield current
                current = nxt

        yield current

    result = ' '.join(join_punctuation([token.text + ('<{}>'.format(token.pos_ if pos == 'pos' else token.tag_ if pos == 'tag' else token.pos_ + ' ' + token.tag_) if not token.pos_ in ('PUNCT', 'SPACE')  else '') for token in doc]))
    result = re.sub(r'\n\ +', '\n', result)
    return result

def aggregate(data):

    # data.loc[:, 'sentiment'] = pd.Series(get_ft_label('\n'.join(data.loc[:, 'text'].apply(lambda x: x.replace('\n', '')))).split('\n'), index=data.index).str[9].astype(int) - 1

    parts_of_speech = reduce(lambda x, y: x.append(y, sort=True), data.loc[:, 'doc'].apply(lambda x: pd.DataFrame({k: [v] for k, v in dict(Counter([token.pos_ for token in x])).items()})))
    parts_of_speech.index = data.index
    parts_of_speech = parts_of_speech.fillna(0)
    data = data.join(parts_of_speech)

    tags = reduce(lambda x, y: x.append(y, sort=True), data.loc[:, 'doc'].apply(lambda x: pd.DataFrame({k: [v] for k, v in dict(Counter([token.tag_ for token in x])).items()})))
    tags.index = data.index
    tags = tags.fillna(0)
    data = data.join(tags, rsuffix='_TAG')

    both = reduce(lambda x, y: x.append(y, sort=True), data.loc[:, 'doc'].apply(lambda x: pd.DataFrame({k: [v] for k, v in dict(Counter([token.pos_ + ' ' + token.tag_ for token in x])).items()})))
    both.index = data.index
    both = both.fillna(0)
    data = data.join(both, rsuffix='_BOTH')

    features = reduce(lambda x, y: x.append(y, sort=True), data.loc[:, 'doc'].apply(lambda x: pd.DataFrame({k: [v] for k, v in dict(Counter([token._.feature for token in x if token._.feature])).items()})))
    features.index = data.index
    features = features.fillna(0)
    data = data.join(features)

    return data

def process_data(data, process_text=True, apply_nlp=True, text_with_features=True, to_aggregate=True):

    data = data.copy()

    data.loc[:, 'text'] = data.loc[:, 'raw text'].str.replace(r'(\.|\,)\w+', lambda x: '{} {}'.format(x.group(0)[0], x.group(0)[1:]), regex=True)\
                                                 .str.replace(r'(\[(.*?)\])|(\((.*?)\))', ' ', regex=True)\
                                                 .str.replace(r'\[|\]|\(|\)', ' ', regex=True)\
                                                 .str.replace(re.escape('..'), ' .SHORT ', regex=True)\
                                                 .str.replace(r'…', ' .LONG ', regex=True)\
                                                 .str.replace(r'^\s*$', '', regex=True)\
                                                 .str.replace(r'\n\s*', '\n', regex=True)\
                                                 .str.replace(r'(?<!\w)((E|e)(M|m|H|h)+)|(U|u|M|m|H|h){2,}(?=\s|,|\.|\-)', '.UMUH', regex=True)\
                                                 .str.replace(r'\’', '\'', regex=True)\
                                                 .str.replace(r'\ +', ' ', regex=True)\
                                                 .str.replace(r'\ +(?=\n)', '', regex=True)

    data.loc[data['text'].str.startswith('\n'), 'text'] = data.loc[data['text'].str.startswith('\n'), 'text'].str.replace(r'\n+', '', 1, regex=True)

    if process_text:
        data.loc[:, 'processed text'] = data.loc[:, 'text'].str.replace(r'((\w+)(-)(\s*))+(\w+)', lambda x: x.group(5) if x.group(2) in x.group(5) else x.group(2) + x.group(5) if x.group(2) + x.group(5) in WORD_SET else x.group(0), regex=True)\
                                                           .str.replace(r'(?<!\w)((\w)\2+)(\w+)', lambda x: x.group(2) + x.group(3) if x.group(2) + x.group(3) in WORD_SET else x.group(0), regex=True)\
                                                           .str.replace(r'(\w+)(-)', '', regex=True)\
                                                           .str.replace(r'\b((\w|\')+)((\s|\,|\.)+\1)+\b', lambda x: x.group(1), regex=True)\
                                                           .str.replace(r'\.[A-Z]+', '', regex=True)\
                                                           .str.replace(r'\ +', ' ', regex=True)\
                                                           .str.replace(r'(?<![.])(?=[\n\r]|$)', '.', regex=True)\
                                                           .str.replace(r'(\,|\.)(\s*(\,|\.))+', lambda x: x.group(1), regex=True)\
                                                           .str.replace(r'\s+\.', '.', regex=True)

    if apply_nlp:
        data.loc[:, 'doc'] = data.loc[:, 'processed text'].apply(nlp) if 'processed text' in data.columns else data.loc[:, 'text'].apply(nlp)
        data.loc[:, 'doc'].apply(add_feature)

        if text_with_features:
            data.loc[:, 'text with features'] = data.loc[:, 'doc'].apply(get_text_with_features)

        if to_aggregate:
            data = aggregate(data)

    return data

def graph_pos(data, pidn, features=[], sentiment=False):

    if features:
        freq = data.loc[pidn, features].T.apply(lambda x: x / x.sum()).T
    else:
        freq = data.loc[pidn, 'ADJ':'VERB'].T.apply(lambda x: x / x.sum()).T

    if sentiment:
        fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [9, 1]}, figsize=(12, 6))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        axes = [ax]

    sns.barplot(x='variable',
                y='value',
                data=pd.melt(freq.reset_index(), id_vars='Number'),
                ax=axes[0])

    axes[0].set_xlabel('Part of Speech')
    axes[0].set_ylabel('Frequency')

    if sentiment:
        sns.barplot(y=data.loc[pidn, 'sentiment'], ax=axes[1])

    fig.suptitle('PIDN={}, n={}'.format(pidn, data.loc[pidn].shape[0]))

    return fig

def graph_features(data, variant, features=[], sentiment=False):
    
    data = data.loc[data['variant'] == variant]

    if features:
        freq = data.loc[:, features].T.apply(lambda x: x / x.sum()).T
        figsize = (len(features), 4)
        width_ratios = (len(features), 1)
    else:
        freq = data.loc[:, 'ADJ':'VERB'].T.apply(lambda x: x / x.sum()).T
        figsize = (12, 4)
        width_ratios = (12, 1)

    if sentiment:
        figsize = (figsize[0] + 4, figsize[1])
        fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': width_ratios, 'wspace': 0.5}, figsize=figsize)
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        axes = [ax]

    sns.barplot(x='variable',
                y='value',
                data=pd.melt(freq.reset_index(), id_vars='pidn'),
                ax=axes[0])

    axes[0].set_xlabel('Feature')
    axes[0].set_ylabel('Frequency')
    axes[0].set_ylim((0, 0.6))

    if sentiment:
        sns.barplot(y=data.loc[:, 'sentiment'], ax=axes[1])

    fig.suptitle('variant={}, n={}'.format(variant, data.shape[0]))

    return fig
