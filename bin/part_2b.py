#!/usr/bin/env python
"""
coding=utf-8

Code supporting Part 2B of Capital One Data Science challenge.

This section of the code challenge centers around a choose-your-adventure type question based in the infamous baby
names data set


"""
import glob
import logging
import os

from collections import Counter

import pandas as pd
from nltk.corpus import words

logging.basicConfig(level=logging.DEBUG)

def main():
    """
    Run code for Part 2B analysis
    :return: None
    :rtype: None
    """
    # Create Dataset
    names_df = baby_names_etl('../data/input/namesbystate')

    # Create aggregate
    print create_year_aggregates(names_df)


def baby_names_etl(data_path):
    """
    Perform basic ETL on baby names data

    :param data_path: path to data, consistent w/ data from http://www.ssa.gov/oact/babynames/state/namesbystate.zip
    :return:
    """
    # Constants
    header_list = ['state_abbr', 'gender', 'birth_year', 'name', 'num_occurrences']

    # Read in data from file
    glob_pattern = os.path.join(data_path, '*.TXT')
    logging.info('Glob pattern for input data: %s' % glob_pattern)
    all_files = glob.glob(glob_pattern)
    logging.info('List of data files: %s' % (all_files))
    names_df = pd.concat(pd.read_csv(filepath_or_buffer=f, sep=',', names=header_list) for f in all_files)

    # Feature engineering
    names_df['male'] = names_df['gender'] == 'M'

    # Constants for later use
    vowels = set(['a', 'e', 'i', 'o', 'u', 'y'])
    consonants = set(['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'x', 'z', 'w'])
    words_normalized = words.words()
    words_normalized = map(lambda x: x.lower(), words_normalized)
    words_normalized = set(words_normalized)

    # Create unique names DF to reduce run time
    unique_names_df = pd.DataFrame(names_df['name'].unique())
    unique_names_df.columns = ['name']
    unique_names_df['name_length'] = unique_names_df['name'].apply(lambda x: len(x))
    unique_names_df['name_counter'] = unique_names_df['name'].apply(lambda x: Counter(x.lower()))
    unique_names_df['num_consonants'] = unique_names_df['name_counter'].apply(lambda x: sum(x[c] for c in consonants))
    unique_names_df['num_vowels'] = unique_names_df['name_counter'].apply(lambda x: sum(x[c] for c in vowels))
    unique_names_df['num_non_letter'] = unique_names_df['name_length'] - (unique_names_df['num_consonants'] + unique_names_df['num_vowels'])
    unique_names_df['vowel_percentage'] = unique_names_df['num_vowels'] / unique_names_df['name_length']
    unique_names_df['name_is_word'] = unique_names_df['name'].apply(lambda x: x.lower() in words_normalized)

    # Drop temporary columns
    unique_names_df = unique_names_df.drop('name_counter', axis=1)

    # Merge unique names back in
    names_df = pd.merge(left=names_df, right=unique_names_df, on='name', how='left')
    names_df.to_csv('../data/output/part2/p2b.csv', index=False)

    # Output
    logging.info('Data description: \n%s' % names_df.describe(include='all'))
    return names_df

def wavg(group, avg_name, weight_name):
    """

    Weighted average function, borrowed from:
     - http://stackoverflow.com/questions/10951341/pandas-dataframe-aggregate-function-using-multiple-columns

    """
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / float(w.sum())
    except ZeroDivisionError:
        return d.mean()

def create_year_aggregates(names_df):
    """
    Aggregate data by year, and compute mean statistics weighted w/ num_occurenences.
    :param names_df: Names dataframe
    :type names_df: pd.DataFrame
    :return: Aggregated dataframe
    :rtype: pd.DataFrame
    """

    # Create aggregate frame
    agg_frame = pd.DataFrame()

    # Add statistics
    agg_frame['avg_name_length'] = names_df.groupby(by=['birth_year']).apply(wavg, 'name_length', 'num_occurrences')
    agg_frame['avg_num_consonants'] = names_df.groupby(by=['birth_year']).apply(wavg, 'num_consonants', 'num_occurrences')
    agg_frame['avg_num_vowels'] = names_df.groupby(by=['birth_year']).apply(wavg, 'num_vowels', 'num_occurrences')
    agg_frame['avg_num_non_letter'] = names_df.groupby(by=['birth_year']).apply(wavg, 'num_non_letter', 'num_occurrences')
    agg_frame['avg_vowel_percentage'] = names_df.groupby(by=['birth_year']).apply(wavg, 'vowel_percentage', 'num_occurrences')
    agg_frame['avg_name_is_word'] = names_df.groupby(by=['birth_year']).apply(wavg, 'name_is_word', 'num_occurrences')

    # Output
    agg_frame.to_csv('../data/output/part2/p2b_year_agg.csv', index=True)
    return agg_frame


# Main section
if __name__ == '__main__':
    main()
