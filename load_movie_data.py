# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:14:07 2020

@author: nicol
"""

import re
import pandas as pd
import numpy as np

filepath = 'C:\\Users\\nicol\\Google Drive\\Datasets\\MovieLens\\'

def load_movie_data():
    # Load csv files
    filepath = 'C:\\Users\\nicol\\Google Drive\\Datasets\\MovieLens\\'
    movies = pd.read_csv(filepath + 'movies.csv')
    ratings = pd.read_csv(filepath + 'ratings.csv')
    
    # Extract genres
    movies = extract_genres(movies)
    
    # Compile ratings
    movies = compile_ratings(movies, ratings)
    
    return movies

def genre_list(df):

    # Extract genres from list
    genre_lists = pd.Series([x.split('|') for x in df['genres']])
    genre_flat_list = set([item for sublist in genre_lists for item in sublist])

    return genre_flat_list

def extract_genres(df):

    # Extract genres from list
    genre_lists = pd.Series([x.split('|') for x in df['genres']])

    # Loop through genres and see which movies have genre listed
    for genre in genre_list(df):
        df[genre] = pd.Series([genre in x for x in genre_lists]).astype(int)
    
    return df

def compile_ratings(movies_df, ratings_df):
    """ Compute weighted ratings according to IMDB's rules
    """
    
    # Compute mean rating, number of ratings, mean score across all ratings
    mean_rating = ratings_df.groupby('movieId')['rating'].mean() / 5 # Scale 0-1
    num_ratings = ratings_df.groupby('movieId')['rating'].count()
    mean_across_movies = np.mean(ratings_df['rating']) / 5 # Scale 0-1
    
    # Minimum number of ratings needed to be considered
    m = 10
    
    # Compute weighted rating
    weighted_rating = (num_ratings / (num_ratings + m) * mean_rating) + (m / (num_ratings + m) * mean_across_movies)
    
    # Put values together in dataframe
    rating_summary = mean_rating.reset_index()
    rating_summary = rating_summary.merge(num_ratings, how='inner', left_on='movieId', right_on='movieId')
    rating_summary = rating_summary.merge(weighted_rating, how='inner', left_on='movieId', right_on='movieId')
    # Set column names
    rating_summary.columns = ['movieId', 'mean_rating', 'num_ratings', 'weighted_rating']
    
    # Add rating dataframe to movies dataframe
    # Outer merge to be safe for now, might change to inner merge if there's no use for movies without ratings
    movies_df = movies_df.merge(rating_summary, how='outer', left_on='movieId', right_on='movieId')
    
    return movies_df
