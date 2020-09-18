# -*- coding: utf-8 -*-
"""
Search movie data, filtering by genres and tags, and return highest-rated movies.

    @author: Nicole Peltier
    @contact: nicole.eleni.peltier@gmail.com
    @date: September 18, 2020
"""

def filter_genre(movies_df, genres):
    """
    Filter rows of dataframe, return only instances of given genres.

    Parameters:
        movies_df (pd DataFrame): dataframe of movie information
        genres (str or list of str): genres by which to filter movies (e.g.,
                        'Comedy' or ['Comedy', 'Romance'])

    Returns:
        df_filtered (pd DataFrame): select rows of input dataframe that are of
                        the given genres
    """

    # If just one genre given, simple filter
    if isinstance(genres, str):
        return movies_df[movies_df[genres.capitalize()]==1]

    # If list of genres, filter iteratively
    else:
        df_filtered = movies_df.copy()
        # Loop through genres in list
        for g in genres:
            df_filtered = df_filtered[df_filtered[g.capitalize()]==1]

        return df_filtered

def filter_tag(movies_df, tags):
    """
    Filter rows of dataframe, return only instances of given genres.

    Parameters:
        movies_df (pd DataFrame): dataframe of movie information
        tags (str or list of str): tags by which to filter movies (e.g.,
                        'funny' or ['hughgrant', 'juliaroberts'])

    Returns:
        df_filtered (pd DataFrame): select rows of input dataframe that have
                        given tags
    """

    # If just one tag given, simple filter
    if isinstance(tags, str):
        return movies_df[movies_df['tag_soup'].str.contains(tags)]

    # If list of genres, filter iteratively
    else:
        df_filtered = movies_df.copy()
        # Loop through tags in list
        for t in tags:
            df_filtered = df_filtered[df_filtered['tag_soup'].str.contains(t)]

        return df_filtered

def highest_rated(movies_df, genres=None, tags=None):
    """
    Find highest rated movies of given genres and tags

    Parameters:
        movies_df (pd DataFrame): dataframe of movie information
        genres (str or list of str): genres by which to filter movies (e.g.,
                        'Comedy' or ['Comedy', 'Romance'])
        tags (str or list of str): tags by which to filter movies (e.g.,
                        'funny' or ['hughgrant', 'juliaroberts'])

    Returns:
        pd.Series of titles of highest rated moves of given genres and tags
    """

    # If user provides genres, filter data by genre
    if genres is not None:
        movies_df = filter_genre(movies_df, genres)

    # If user provides tags, filter data by tag
    if tags is not None:
        movies_df = filter_tag(movies_df, tags)

    # Sort movies by rating
    df_sorted = movies_df.sort_values(by=['weighted_rating'], ascending=False)
    num_movies = df_sorted.shape[0]

    # If fewer than 5 movies, return all
    if num_movies < 5:
        return df_sorted['title']

    # If at least 5 movies, return 5
    else:
        return df_sorted.iloc[:5]['title']
