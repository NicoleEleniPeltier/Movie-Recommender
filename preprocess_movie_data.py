# -*- coding: utf-8 -*-
"""
Load data from MovieLens dataset (saved locally) and perform preprocessing to
create a movie recommender system.

    @author: Nicole Peltier
    @contact: nicole.eleni.peltier@gmail.com
    @date: September 17, 2020
"""

import pandas as pd
import numpy as np

FILEPATH = 'C:\\Users\\nicol\\Google Drive\\Datasets\\MovieLens\\'
FILEPATH_LARGE = 'C:\\Users\\nicol\\Google Drive\\Datasets\\MovieLens-Large\\'

def load_movie_data():
    """
    Load csv files containing movie information and rating information.
    Preprocesses data for analysis by a movie recommender system.

    Returns:
        movies (pd DataFrame): dataframe containing movie title, genre
                        information, and rating information
    """

    # Load csv files
    movies = pd.read_csv(FILEPATH + 'movies.csv')
    ratings = pd.read_csv(FILEPATH_LARGE + 'ratings.csv')

    # Extract genres
    movies = extract_genres(movies)

    # Compile ratings
    movies = compile_ratings(movies, ratings)

    # Load tags and create tag soup
    movies = load_tags(movies)

    # Remove year from title and add year column
    movies['year'] = movies['title'].str.extract(r'[(]([0-9]{4})[)]')
    movies['title'] = movies['title'].str.replace(r' [(][0-9]{4}.*[)]', '')

    # Remove 13 rows with NaN for year, convert year to int
    movies = movies.dropna()
    movies.loc[:, 'year'] = movies['year'].astype(int)

    # Store cleaned version of title with lowercase letters and no punctuation
    movies['title_clean'] = movies['title'].apply(clean_title)

    return movies

def clean_title(string):
    """
    Clean movie title to make user searches more fool-proof. Convert to
    lowercase, remove "the" and "a" from start of title, and remove punctuation.

    Parameters:
        string (str): movie title

    Returns:
        title_clean (str): cleaned movie title
    """

    # Remove endings of strings that end with ", The" and ", A"
    title_clean = string.replace(r', The$', '')
    title_clean = title_clean.replace(r', A$', '')

    # Remove starts of strings that begin with "A " and "The "
    title_clean = title_clean.replace(r'^The ', '')
    title_clean = title_clean.replace(r'^A ', '')

    # Make strings lowercase
    title_clean = title_clean.lower()

    # Remove/replace punctuation
    chars_to_replace = ".,:'?!"
    for ch in chars_to_replace:
        title_clean = title_clean.replace(ch, '')
    title_clean = title_clean.replace(' -', '')
    title_clean = title_clean.replace('-', ' ')

    return title_clean

def genre_list(df):
    """
    Generate list of genres from dataframe.

    Parameters:
        df (pd DataFrame): dataframe containing genres listed for each movie

    Returns:
        genre_flat_list (set): set of unique genres that appear in dataframe
    """

    # Each movie's genre is initially coded as "Comedy|Romance|Animation"
    # Convert each movie's genre to a list: ["Comedy", "Romance", "Animation"]
    genre_lists = pd.Series([x.split('|') for x in df['genres']])

    # Find unique occurrences across all movies
    genre_flat_list = set([item for sublist in genre_lists for item in sublist])

    return genre_flat_list

def extract_genres(df):
    """
    Convert each movie's genres to a one-hot representation.

    Parameters:
        df (pd DataFrame): dataframe of movie information

    Returns:
        df (pd DataFrame): updated dataframe with one column for each genre;
                        samples that were labeled as a given genre have 1 in
                        that genre's column, otherwise 0
    """

    # Extract genres from list
    genre_lists = pd.Series([x.split('|') for x in df['genres']])

    # Loop through genres and see which movies have genre listed
    for genre in genre_list(df):
        df[genre] = pd.Series([genre in x for x in genre_lists]).astype(int)

    return df

def compile_ratings(movies_df, ratings_df):
    """
    Compute weighted ratings according to IMDB's formula.

    Parameters:
        movies_df (pd DataFrame): dataframe of movie information
        ratings_df (pd DataFrame): dataframe of rating information

    Returns:
        movies_df (pd DataFrame): updated movie dataframe with rating information added
    """

    # Compute mean rating, number of ratings, mean score across all ratings
    mean_rating = ratings_df.groupby('movieId')['rating'].mean() / 5 # Scale 0-1
    num_ratings = ratings_df.groupby('movieId')['rating'].count()
    mean_across_movies = np.mean(ratings_df['rating']) / 5 # Scale 0-1

    # Minimum number of ratings needed to be considered
    m = 200

    # Compute weighted rating
    weighted_rating = ((num_ratings / (num_ratings + m) * mean_rating) +
                       (m / (num_ratings + m) * mean_across_movies))

    # Put values together in dataframe
    rating_summary = mean_rating.reset_index()
    rating_summary = rating_summary.merge(num_ratings, how='inner',
                                          left_on='movieId', right_on='movieId')
    rating_summary = rating_summary.merge(weighted_rating, how='inner',
                                          left_on='movieId', right_on='movieId')

    # Set column names
    rating_summary.columns = ['movieId', 'mean_rating', 'num_ratings', 'weighted_rating']

    # Add rating dataframe to movies dataframe
    # Outer merge to be safe for now, might change to inner merge if there's
    # no use for movies without ratings
    movies_df = movies_df.merge(rating_summary, how='left',
                                left_on='movieId', right_on='movieId')

    return movies_df

def process_tag(string):
    """
    Convert user-written tags to lowercase strings without punctuation or
    spaces. Function allows list comprehension of tags in load_tags().

    Parameters:
        string (str): tag written by user

    Returns:
        string (str): updated tag with punctuation and spaces removed
    """

    # Convert string to lowercase
    string = string.lower()

    # List of characters to replace in string
    chars_to_replace = " &-!,.?'():;*"

    # Loop through characters to be replaced
    for ch in chars_to_replace:
        string = string.replace(ch, '')

    return string

def load_tags(movies_df):
    """
    Load tags assigned by users and pool them within movies.

    Parameters:
        movies_df (pd DataFrame): dataframe of movie information

    Returns:
        movies_df (pd DataFrame): updated dataframe with new column 'tag_soup'
    """

    # Load tags from larger dataset
    tags = pd.read_csv(FILEPATH_LARGE + 'tags.csv')

    # Remove punctuation
    tags['tag_soup'] = [process_tag(str(x)) for x in tags['tag']]

    # Join all tags for same movie
    tag_soup = tags.groupby(['movieId'])['tag_soup'].apply(' '.join).reset_index()

    # Merge tag soup with movies df
    movies_df = movies_df.merge(tag_soup, how='inner', left_on='movieId', right_on='movieId')

    # Replace nan with ''
    movies_df['tag_soup'] = movies_df['tag_soup'].fillna('')

    return movies_df

def main():
    """
    Main function of module. Load movie data, process ratings and user tags,
    save preprocessed data.
    """
    movie_data = load_movie_data()

    # Save preprocessed data
    movie_data.to_csv('movies_preprocessed.csv', index=False)

if __name__ == "__main__":
    main()
