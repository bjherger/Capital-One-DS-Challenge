#!/usr/bin/env python
"""
coding=utf-8

Code Template

"""
import glob
import logging
import os

logging.basicConfig(level=logging.DEBUG)

import pandas as pd


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

    # Output
    logging.info('Data description: \n%s' % names_df.describe(include='all'))
    return names_df


def p2qa3(names_df):
    """
    Part 2, question A2:
     - What is the most gender ambiguous name in 2013? 1945?
    :param names_df:
    :return:
    """

    # Create year / threshold permutations
    result_df = pd.DataFrame(
        [{'birth_year': 2013, 'min_num_observations': 0},
         {'birth_year': 2013, 'min_num_observations': 10},
         {'birth_year': 2013, 'min_num_observations': 100},
         {'birth_year': 2013, 'min_num_observations': 1000},
         {'birth_year': 2013, 'min_num_observations': 10000},
         {'birth_year': 1945, 'min_num_observations': 0},
         {'birth_year': 1945, 'min_num_observations': 10},
         {'birth_year': 1945, 'min_num_observations': 100},
         {'birth_year': 1945, 'min_num_observations': 1000},
         {'birth_year': 1945, 'min_num_observations': 10000}])

    # Compute results
    result_df['most_neutral_name'] = result_df.apply(
        func=lambda x: most_neutral_name(names_df, x['birth_year'], x['min_num_observations']), axis=1)

    result_df.to_csv('../data/output/part2/p2qa2.csv', index=False)
    return result_df


def most_neutral_name(names_df, year, min_num_observations):
    logging.debug('Computing most gender neutral name, with arguments: %s' % locals())
    # Subset to correct year
    names_df = names_df[names_df['birth_year'] == year]

    # Get count across all states
    count_df = names_df[['name', 'gender', 'num_occurrences']].groupby(by=['name', 'gender'], as_index=False).sum()

    # Get gender-specific counts
    male_df = count_df[count_df['gender'] == 'M']
    male_df = male_df.rename(columns={'num_occurrences': 'male_num_occurrences'})
    female_df = count_df[count_df['gender'] == 'F']
    female_df = female_df.rename(columns={'num_occurrences': 'female_num_occurrences'})

    # Create a dataframe w/ number of observations for each name, for each gender
    gender_df = pd.merge(male_df, right=female_df, how='outer', left_on='name', right_on='name')
    gender_df = gender_df[['name', 'male_num_occurrences', 'female_num_occurrences']]
    gender_df = gender_df.fillna(0)

    # Compute features
    gender_df['total_num_occurrences'] = gender_df['male_num_occurrences'] + gender_df['female_num_occurrences']
    gender_df['male_ratio'] = gender_df['male_num_occurrences'] / gender_df['total_num_occurrences']

    # Threshold
    gender_df = gender_df[gender_df['total_num_occurrences'] >= min_num_observations]

    # Compute gender preference (lower is less preference for one gender, higher is more preference
    gender_df['gender_preference'] = gender_df['male_ratio'].apply(lambda x: abs(x - .5))

    # Sort
    gender_df = gender_df.sort_values(by=['gender_preference', 'total_num_occurrences'], ascending=[True, False])

    # Return most neutral name
    return gender_df.iloc[0]['name']


def p2qa4(names_df):
    """
    Part 2, Question A3:
     - Of the names represented in the data, find the name that has had the largest
    percentage increase in popularity since 1980. Largest decrease?

    :param names_df:
    :return:
    """
    year_lower = 1980
    year_higher = 2015

    # Create DF for lower year with name popularity (regardless of gender)
    lower_year_popularity = names_df[names_df['birth_year'] == year_lower]
    lower_year_popularity = lower_year_popularity[['name', 'num_occurrences']].groupby('name', as_index=False).sum()

    # Create DF for higher year with name popularity (regardless of gender)
    higher_year_popularity = names_df[names_df['birth_year'] == year_higher]
    higher_year_popularity = higher_year_popularity[['name', 'num_occurrences']].groupby('name', as_index=False).sum()

    # Combine two dataframes togethr
    count_df = pd.merge(left=lower_year_popularity, right=higher_year_popularity, how='outer', on='name',
                        suffixes=('_' + str(year_lower), '_' + str(year_higher)))
    count_df = count_df.fillna(0)

    # Feature engineering
    count_df['perc_change'] = (count_df['num_occurrences_2015'] - count_df['num_occurrences_1980']) / count_df[
        'num_occurrences_1980']

    results_df = pd.DataFrame(
        [{'ordering': 'increase', 'min_num_observations': 0},
         {'ordering': 'increase', 'min_num_observations': 10},
         {'ordering': 'increase', 'min_num_observations': 100},
         {'ordering': 'increase', 'min_num_observations': 1000},
         {'ordering': 'increase', 'min_num_observations': 10000},
         {'ordering': 'decrease', 'min_num_observations': 0},
         {'ordering': 'decrease', 'min_num_observations': 10},
         {'ordering': 'decrease', 'min_num_observations': 100},
         {'ordering': 'decrease', 'min_num_observations': 1000},
         {'ordering': 'decrease', 'min_num_observations': 10000}])

    # Compute results
    new_cols = results_df.apply(
        lambda x: biggest_change(count_df, ordering=x['ordering'], min_num_observations=x['min_num_observations'])
        , axis=1)
    new_cols.columns = ['biggest_change', 'num_occurrences_1980', 'num_occurrences_2015', 'perc_change']

    # Attach results to results DF
    results_df = results_df.join(other=new_cols)

    # Output
    results_df.to_csv('../data/output/part2/p2qa4.csv', index=False)
    return results_df


def biggest_change(count_df, ordering, min_num_observations):
    # Convert from ordering (ascending or descending) to whether to sort ascending. This is somewhat non-intuitive,
    # so I've abstracted it from the results DF
    ascending = dict(increase=False, decrease=True)[ordering]

    # Subset to relevent number of observations
    count_df = count_df[(count_df['num_occurrences_1980'] >= min_num_observations)
                        & (count_df['num_occurrences_2015'] >= min_num_observations)]

    # Return result
    return count_df.sort_values(by='perc_change', ascending=ascending).iloc[0]


def main():
    """
    Main function documentation template
    :return: None
    :rtype: None
    """
    names_df = baby_names_etl('../data/input/namesbystate')

    # Part 2, Question A2
    # p2qa3_results = p2qa3(names_df)
    # print p2qa3_results

    # p2qa4_results = p2qa4(names_df)
    # print p2qa4_results


# Main section
if __name__ == '__main__':
    main()
