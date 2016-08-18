#!/usr/bin/env python
"""
coding=utf-8

Code Template

"""
import logging
logging.basicConfig(level=logging.DEBUG)

import pandas as pd

def data_etl(data_path):
    logging.info('ETL of data from path: %s' % data_path)

    # Load data
    data_df = pd.read_csv(data_path, sep='\t')
    logging.info('Data description: \n%s' % data_df.describe())
    print data_df.isnull().sum()
    print data_df.dropna().describe()


# Functions
def main():
    """
    Main function documentation template
    :return: None
    :rtype: None
    """
    data_etl('../data/input/codetest_train.txt')

# Main section
if __name__ == '__main__':
    main()
