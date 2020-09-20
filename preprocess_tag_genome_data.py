# -*- coding: utf-8 -*-
"""
Preprocess genome tag data and save in csv file.

    @author: Nicole Peltier
    @contact: nicole.eleni.peltier@gmail.com
    @date: September 19, 2020
"""

import pandas as pd
from preprocess_movie_data import process_tag

def load_genome_tags():
    """
    Load and format genome score and genome tag data.
    """

    # Load csv files with genome scores and tag IDs
    filepath = "C:\\Users\\nicol\\Google Drive\\Datasets\\MovieLens-Large\\"
    genome_scores = pd.read_csv(filepath + "genome-scores.csv")
    genome_tags = pd.read_csv(filepath + "genome-tags.csv")

    # Pivot table to have each tag be one column and each movie be one row
    genome_tag_df = genome_scores.pivot(index='movieId', columns='tagId', values='relevance')

    # Set column names to be tag names
    genome_tag_df.columns = genome_tags['tag'].values

    return genome_tag_df

def get_relevant_tag_soup(df):
    """
    Generate soup of tags for each movie with relevance scores of at least 0.75.

    Parameters:
        df (pd DataFrame): dataframe of relevance score for each tag

    Returns:
        df (pd DataFrame): updated dataframe with new column 'relevant_tag_soup'
    """

    soup = []

    # Loop through movies, collect tags with relevance >= 0.75
    for i in range(df.shape[0]):
        x = df.loc[i].squeeze().sort_values(ascending=False)
        relevant_tags = list(x[x>=0.75].index)
        # Make tags lowercase, remove punctuation
        tag_list = [process_tag(x) for x in relevant_tags]
        # Join list of tags into soup
        relevant_tag_soup = ' '.join(tag_list)
        soup.append(relevant_tag_soup)

    return soup

def main():
    """
    Main function of module. Load genome data, merge with movie dataframe,
    save preprocessed data.
    """

    # Load genome tags
    filepath = 'C:\\Users\\nicol\\OneDrive\\Documents\\AI Project\\'
    genome_tag_df = load_genome_tags()

    # Load main movie dataframe
    movie_df = pd.read_csv(filepath + 'movies_preprocessed.csv')

    # Select just movieId and title_clean to match rows of movie dataframe with
    # rows of tag dataframe
    movie_id_title = movie_df[['movieId', 'title_clean']]
    movie_genome_tags = movie_id_title.merge(genome_tag_df.reset_index(), how='left', left_on='movieId', right_on='movieId')

    # Create soup of for each movie of tags with relevance of at least 0.75
    tag_names = genome_tag_df.columns.values
    movie_genome_tags['relevant_tag_soup'] = get_relevant_tag_soup(movie_genome_tags[tag_names])

    # Save preprocessed data
    movie_genome_tags.to_csv('tag_genome_preprocessed.csv', index=False)

if __name__ == "__main__":
    main()
