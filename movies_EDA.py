# -*- coding: utf-8 -*-
"""
Plot features of MovieLens dataset for exploratory data analysis.

    @author: Nicole Peltier
    @contact: nicole.eleni.peltier@gmail.com
    @date: September 17, 2020
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from load_movie_data import genre_list

def plot_genre_dists(df):
    """
    Plot number of occurrences of each genre across dataset.

    Parameters:
        df (pd DataFrame): dataframe containing movie title, genre
                        information, and rating information

    Returns:
        None
    """

    # Get list of genres
    genres = list(genre_list(df))
    # Select genre columns from movie dataframe
    genre_df = df[genres]
    # Count occurrences of each genre
    genre_count = genre_df.sum().reset_index()
    # Sort genres by number of occurrences (to make plot look neater)
    genre_count.sort_values(by=[0], ascending=False, inplace=True)

    # Plot occurrences of each genre
    plt.figure(figsize=(10,4))
    genre_plot = sns.barplot(genre_count['index'], genre_count[0])
    # Modify figure appearance
    plt.title('Genre count in movie dataset')
    plt.xlabel('Genre')
    plt.ylabel('Count')
    for item in genre_plot.get_xticklabels():
        item.set_rotation(75)

def plot_genre_count_per_movie(df):
    """
    Plot distribution of number of genres assigned to each movie in dataset.

    Parameters:
        df (pd DataFrame): dataframe containing movie title, genre
                        information, and rating information

    Returns:
        None
    """

    # Get list of genres
    genres = list(genre_list(df))
    # Select genre columns from movie dataframe
    genre_df = df[genres]
    # Count number of genres for each movie
    genre_count = genre_df.sum(axis=1).value_counts()

    # Plot number of movies with different numbers of genres assigned
    plt.figure()
    sns.barplot(genre_count.index, genre_count)
    # Modify figure appearance
    plt.title('Number of genres per movie')
    plt.xlabel('Number of genres')
    plt.ylabel('Count')

def plot_weighted_vs_raw_rating(df):
    """
    Create scatterplot of weighted ratings (per IMDB's formula) vs. mean of
    raw ratings.

    Parameters:
        df (pd DataFrame): dataframe containing movie title, genre
                        information, and rating information

    Returns:
        None
    """

    # Plot weighted vs. raw scores
    plt.figure(figsize=(8,8))
    plt.plot([0, 1], [0, 1], color='#000000') # reference line of unity
    plt.scatter(df['mean_rating'], df['weighted_rating'], c=np.log10(df['num_ratings']), alpha=0.2)

    # Modify figure appearance
    plt.axis('square')
    plt.xlabel('Mean rating')
    plt.ylabel('Weighted rating')
    plt.title('Weighted rating vs. mean raw rating')
    cbar = plt.colorbar()
    cbar.set_alpha(1)
    cbar.draw_all()
    cbar.ax.set_ylabel('# of ratings (log10)', rotation=270);

def plot_rating_histograms(df):
    """
    Plot histograms of mean raw ratings and weighted ratings (per IMDB's formula).

    Parameters:
        df (pd DataFrame): dataframe containing movie title, genre
                        information, and rating information

    Returns:
        None
    """

    # Create figure
    plt.figure(figsize=(12, 4))

    # Subplot 1: raw ratings
    plt.subplot(1, 2, 1)
    sns.distplot(df['mean_rating'], kde=False)
    # Modify figure appearance
    plt.xlim((0, 1))
    plt.title('Distribution of mean raw rating')
    plt.xlabel('Mean rating')
    plt.ylabel('Count')

    # Subplot 2: weighted ratings
    plt.subplot(1, 2, 2)
    sns.distplot(df['weighted_rating'], kde=False)
    # Modify figure appearance
    plt.xlim((0, 1))
    plt.title('Distribution of weighted ratings')
    plt.xlabel('Weighted rating')
    plt.ylabel('Count');