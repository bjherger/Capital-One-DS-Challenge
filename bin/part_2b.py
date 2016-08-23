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

import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

logging.basicConfig(level=logging.DEBUG)


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

    wordnet_lemmatizer = WordNetLemmatizer()
    names_df['lemmatized_name'] = names_df['name'].apply(wordnet_lemmatizer.lemmatize)

    porter_stemmer = PorterStemmer()
    names_df['stemmed_name'] = names_df['name'].apply(porter_stemmer.stem)
    print porter_stemmer.stem('Franklin')
    print porter_stemmer.stem('Frankie')
    print porter_stemmer.stem('Franky')
    print porter_stemmer.stem('Frank')
    # Output
    logging.info('Data description: \n%s' % names_df.describe(include='all'))
    return names_df


def main():
    """
    Main function documentation template
    :return: None
    :rtype: None
    """
    nltk.download('wordnet')
    names_df = baby_names_etl('../data/input/namesbystate')
    print names_df[['name', 'lemmatized_name', 'stemmed_name']]

# Main section
if __name__ == '__main__':
    main()
