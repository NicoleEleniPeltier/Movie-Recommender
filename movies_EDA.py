# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:51:56 2020

@author: nicol
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from load_movie_data import genre_list

def plot_genre_dists(df):
    genres = list(genre_list(df))
    genre_df = df[genres]
    genre_count = genre_df.sum().reset_index()
    genre_count.sort_values(by=[0], ascending=False, inplace=True)
    
    plt.figure(figsize=(10,4))
    genre_plot = sns.barplot(genre_count['index'], genre_count[0])
    plt.title('Genre count in movie dataset')
    plt.xlabel('Genre')
    plt.ylabel('Count')
    for item in genre_plot.get_xticklabels():
        item.set_rotation(75)

def plot_genre_count_per_movie(df):
    genres = list(genre_list(df))
    genre_df = df[genres]
    genre_count = genre_df.sum(axis=1).value_counts()
    
    plt.figure()
    sns.barplot(genre_count.index, genre_count)
    plt.title('Number of genres per movie')
    plt.xlabel('Number of genres')
    plt.ylabel('Count')

def plot_weighted_vs_raw_rating(df):
    plt.figure(figsize=(8,8))
    plt.plot([0, 1], [0, 1], color='#000000')
    plt.scatter(df['mean_rating'], df['weighted_rating'], c=np.log10(df['num_ratings']), alpha=0.2)
    plt.axis('square')
    plt.xlabel('Mean rating')
    plt.ylabel('Weighted rating')
    plt.title('Weighted rating vs. mean raw rating')
    cbar = plt.colorbar()
    cbar.set_alpha(1)
    cbar.draw_all()
    cbar.ax.set_ylabel('# of ratings (log10)', rotation=270);

def plot_rating_histograms(df):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    sns.distplot(df['mean_rating'], kde=False)
    plt.xlim((0, 1))
    plt.title('Distribution of mean raw rating')
    plt.xlabel('Mean rating')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    sns.distplot(df['weighted_rating'], kde=False)
    plt.xlim((0, 1))
    plt.title('Distribution of weighted ratings')
    plt.xlabel('Weighted rating')
    plt.ylabel('Count');