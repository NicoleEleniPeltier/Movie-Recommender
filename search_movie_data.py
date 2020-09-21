# -*- coding: utf-8 -*-
"""
Search movie data, filtering by genres and tags, and return highest-rated movies.

    @author: Nicole Peltier
    @contact: nicole.eleni.peltier@gmail.com
    @date: September 18, 2020
"""

import numpy as np
from preprocess_movie_data import clean_title

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
        return movies_df[movies_df[genres.capitalize()] == 1]

    # If list of genres, filter iteratively
    df_filtered = movies_df.copy()
    # Loop through genres in list
    for g in genres:
        df_filtered = df_filtered[df_filtered[g.capitalize()] == 1]

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
    return df_sorted.iloc[:5]['title']

def levenshtein_ratio(s, t):
    """
    Compute Levenshtein ratio between user input and titles in movie dataframe.

    Parameters:
        s (str): movie title in dataframe
        t (str): movie title input by user

    Returns:
        lev_ratio (float): Levenshtein distance ratio of similarity between s and t
    """

    # Initialize matrix of zeros
    rows = len(s) + 1
    cols = len(t) + 1
    distance = np.zeros((rows, cols), dtype=int)

    # Populate matrix with indices of each character of both strings
    distance[0, :] = np.arange(cols) # First row: 0, 1, ..., cols-1
    distance[:, 0] = np.arange(rows) # First column: 0, 1, ..., rows-1

    # Iterate over the matrix to compute the cost of deletions, insertions, and substitutions
    for col in range(1, cols):
        for row in range(1, rows):
            # If characters match in position [i,j], cost = 0
            if s[row - 1] == t[col - 1]:
                cost = 0
            # If characters do not match, cost = 2
            else:
                cost = 2
            distance[row][col] = min(distance[row - 1][col] + 1, # Cost of deletions
                                     distance[row][col - 1] + 1, # Cost of insertions
                                     distance[row - 1][col - 1] + cost) # Cost of substitutions

    # Compute Levenshtein distance ratio
    lev_ratio = ((len(s) + len(t)) - distance[row][col]) / (len(s) + len(t))
    return lev_ratio

def search_title(movies_df, s):
    """
    Search movie dataframe for matching title. If there is no exact match,
    fuzzy string matching is attempted. Note: movie dataframe can be filtered
    by genre or tag before searching by title to save computation time.

    Parameters:
        movies_df (pd DataFrame): dataframe of movie information
        s (str): title to search

    Returns:
        index (int) of movie with matching title (if there is a match)
    """

    # First, search for exact match between user input and movie titles
    movie_ind = movies_df.index[movies_df['title_clean'] == clean_title(s)]
    num_matches = movie_ind.shape[0]

    # If there is exactly one match, return index
    if num_matches == 1:
        return movie_ind[0]

    # If there is more than one exact match, ask user to select one
    if num_matches > 1:
        print('More than one movie with matching title\n')
        ind_dict = dict(list(zip(np.arange(num_matches)+1, movie_ind.values)))
        for key, val in ind_dict.items():
            print(f"{key}: {movies_df['title'][val]} ({movies_df['year'][val]})")
        # possible_answers: keys for ind_dict, -1
        possible_answers = [str(k) for k in ind_dict.keys()]
        possible_answers.append('-1')
        # Ask user to select movie
        x = ''
        while x not in possible_answers:
            x = input('\nPlease select movie number, or enter -1 to exit: ')

        # Parse user's choice as int, select movie index
        choice = int(x)
        if choice in ind_dict.keys():
            return ind_dict[choice]

    # If no exact matches, perform fuzzy string matching
    else:
        print('Cannot find that title, looking for similar titles...\n')

        # Compute Levenshtein ratio for all titles
        lev_ratio = movies_df.apply(lambda x: levenshtein_ratio(x['title_clean'],
                                                                clean_title(s)), axis=1)

        # Select titles with Levenshtein ratio greater than threshold
        thresh = 0.7
        lev_ratio_select = lev_ratio[lev_ratio > thresh].reset_index()
        # Dataframe of matching movies
        select_movies = movies_df.loc[lev_ratio_select['index']][['movieId', 'title', 'year']].reset_index()

        num_matches = select_movies.shape[0]

        # If no fuzzy matches, return nothing
        if num_matches == 0:
            print('Sorry, no matches found')
            return None

        # If one fuzzy match, ask user if it is correct
        if num_matches == 1:
            print('One possible match found. Is this the title you want?\n')
            # Print Title (Year) of movie
            print(f"{select_movies['title'][0]} ({select_movies['year'][0]})")

            user_input = ''
            while user_input.lower() not in ['y', 'n']:# Loop until user enters 'y' or 'n'
                user_input = input('Enter y or n: ')

            # If user says it's a match, return index
            if user_input.lower() == 'y':
                return select_movies['index'][0]

            # If user says it isn't a match, return nothing
            print("Sorry I couldn't help")
            return None

        # If multiple fuzzy matches, ask user to select one
        print('More than one potential match found\n')

        # Merge select movies df with lev_ratio_select to sort by lev ratio
        select_movies = select_movies.merge(lev_ratio_select, how='inner',
                                            left_on='index', right_on='index')
        sorted_select_movies = select_movies.sort_values(by=[0], ascending=False).reset_index()

        # Create dictionary of sorted indices
        ind_dict = dict(list(zip(np.arange(num_matches)+1, sorted_select_movies['index'])))

        # Print Title (Year) for each candidate movie
        for key, val in ind_dict.items():
            print(f"{key}: {movies_df['title'][val]} ({movies_df['year'][val]})")

        # Possible user answers: keys for ind_dict (1,2,etc.), -1
        possible_answers = [str(k) for k in ind_dict.keys()]
        possible_answers.append('-1')
        # Ask user to select movie
        x = ''
        while x not in possible_answers:# Loop until valid answer
            x = input('\nPlease select movie number, or enter -1 to exit: ')

        # Parse user's choice as int, select movie index
        choice = int(x)
        if choice in ind_dict.keys():
            return ind_dict[choice]

        # If user says it isn't a match, return nothing
        print("Sorry I couldn't help")
        return None
