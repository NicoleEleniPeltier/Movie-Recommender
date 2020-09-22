# -*- coding: utf-8 -*-
"""
Search movie data, filtering by genres and tags, and return highest-rated movies.

    @author: Nicole Peltier
    @contact: nicole.eleni.peltier@gmail.com
    @date: September 18, 2020
"""

import numpy as np
import pandas as pd
from preprocess_movie_data import clean_title
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

def highest_rated(movies_df, genres=None, tags=None, num_movies=5):
    """
    Find highest rated movies of given genres and tags

    Parameters:
        movies_df (pd DataFrame): dataframe of movie information
        genres (str or list of str): genres by which to filter movies (e.g.,
                        'Comedy' or ['Comedy', 'Romance'])
        tags (str or list of str): tags by which to filter movies (e.g.,
                        'funny' or ['hughgrant', 'juliaroberts'])
        num_movies (int): number of movies to return

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

    # If fewer than 5 movies, return all
    if df_sorted.shape[0] < num_movies:
        return df_sorted['title']

    # If at least 5 movies, return 5
    return df_sorted.iloc[:num_movies]['title']

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
        movieId (int) of selected movie
    """

    # First, search for exact match between user input and movie titles
    movie_ind = movies_df.index[movies_df['title_clean'] == clean_title(s)]
    num_matches = movie_ind.shape[0]

    # If there is exactly one match, return index
    if num_matches == 1:
        return movies_df['movieId'][movie_ind[0]]

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
            return movies_df['movieId'][ind_dict[choice]]

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
                return select_movies['movieId'][0]

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
            return movies_df['movieId'][ind_dict[choice]]

        # If user says it isn't a match, return nothing
        print("Sorry I couldn't help")
        return None

def get_most_relevant_tags(tags_df, title, num_tags=10):
    """
    Get list of most relevant tags for a given movie.

    Parameters:
        tags_df (pd.DataFrame): dataframe containing tag relevance scores for
                        each movie
        title (str): movie title to search
        num_tags (int): number of relevant tags to return, optional (default = 10)

    Returns:
        pd.Series of most relevant tags for movie, index is tag name and value
                        is relevance score
    """

    # Get movie ID that corresponds to title
    movieId = search_title(tags_df, title)

    # Get list of tags
    tag_names = tags_df.drop(['movieId', 'title', 'title_clean', 'year', 'relevant_tag_soup'],
                             axis=1).columns.values
    
    # Slice row with movie
    movie = tags_df[tags_df['movieId']==movieId]

    # Get series of tags, sort in order of relevance to movie
    x = movie[tag_names].squeeze().sort_values(ascending=False)

    # Return series of most relevant tags
    return x[:num_tags]

def get_relevant_tag_soup(movies_df, title):
    """
    Get list of most tags with a relevance score of at least 0.75 for a given movie.

    Parameters:
        movies_df (pd.DataFrame): dataframe containing movie title and tag soup
        title (str): movie title to search

    Returns:
        str containing all relevant tags
    """

    movieId = search_title(movies_df, title)
    return movies_df[movies_df['movieId']==movieId]['relevant_tag_soup'].values[0]

def get_similar_movies(movies_df, title, how='tag_soup', field='tag_soup', num_movies=10):
    """
    Return titles of movies that are similar to a given movie, given their
    user-assigned tags. Note: movies dataframe may be pre-filtered for genre
    or tags.

    Parameters:
        movies_df (pd DataFrame): dataframe of movie information
        title (str): title to search
        field (str): field of movies_df containing tags, optional (default = 'tag_soup')
        num_movies (int): number of results to return, optional (default = 10)

    Returns:
        pd DataFrame of movies most similar to input title
    """

    # Reset index of movies_df in case any filtering of df has already happened
    # and indices are not consecutive
    movies_df = movies_df.reset_index().drop('index', axis=1)

    # Find movie ID based on title
    movieId = search_title(movies_df, title)

    # If no movie could be found, raise error
    if movieId is None:
        raise Exception("Sorry, no movie found. Movie may not be in dataframe or may not match filters.")

    movie_ind = movies_df.index[movies_df['movieId'] == movieId][0]

    if how == 'tag_soup':
        related_ids = get_similar_tag_soup(movies_df, movie_ind, field, num_movies)
    elif how == 'tag_relevance':
        related_ids = get_similar_tag_relevance(movies_df, movieId, num_movies)
    else:
        raise Exception("Invalid method: {}".format(how))

    related_movies = movies_df[movies_df['movieId'].isin(related_ids)]

    return related_movies

def get_similar_titles(movies_df, title, how='tag_soup', field='tag_soup', num_movies=10):
    """
    Create titles of movies that are similar to input title.

    Parameters:
        movies_df (pd DataFrame): dataframe of movie information
        title (str): title to search
        field (str): field of movies_df containing tags, optional (default = 'tag_soup')
        num_movies (int): number of results to return, optional (default = 10)

    Returns:
        pd.Series of titles of movies most similar to input title
    """

    # Get dataframe of similar movies
    related_movies = get_similar_movies(movies_df, title, how, field, num_movies)

    # Return titles
    return related_movies['title']
    

def get_similar_tag_soup(movies_df, movie_ind, field='tag_soup', num_movies=10):
    """
    Find movies that have similar assigned tags to a target movie.

    Parameters:
        movies_df (pd DataFrame): dataframe of movie information
        movie_ind (int): index of target movie in dataframe
        field (str): field of movies_df containing tags, optional (default = 'tag_soup')
        num_movies (int): number of results to return, optional (default = 10)

    Returns:
        pd Series of movieId of most similar movies
    """

    # Tokenize words in tag soup
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(movies_df[field])
    count_matrix_select_movie = count_matrix[movie_ind, :]

    # Compute cosine similarity between target movie and all others
    cos_sim = cosine_similarity(count_matrix, count_matrix_select_movie)

    # Create dataframe of movies and cosine similarity, sort by similarity
    movie_similarity = pd.DataFrame({'movieId': movies_df['movieId'],
                                     'cos_sim': np.squeeze(cos_sim)})
    movie_similarity = movie_similarity.sort_values(by=['cos_sim'], ascending=False)

    # Return top matches
    return movie_similarity['movieId'][1:num_movies+1]

def get_similar_tag_relevance(genome_df, movieId, num_movies=10):
    """
    Find movies that have similar assigned tags to a target movie.

    Parameters:
        movies_df (pd DataFrame): dataframe of movie information
        movieId (int): identification number of target movie
        num_movies (int): number of results to return, optional (default = 10)

    Returns:
        pd Series of movieId of most similar movies
    """

    # Drop rows without tag genome relevance scores
    gen_data = genome_df.dropna().reset_index()

    # Get index of movie in dataframe
    movie_ind = np.squeeze(gen_data.index[gen_data['movieId']==movieId])

    # Convert relevance scores to matrix
    gen_data_matrix = gen_data.drop(['index', 'movieId', 'title', 'title_clean', 'relevant_tag_soup'], axis=1).values
    # Extract row for target movie
    gen_data_select_movie = gen_data_matrix[movie_ind,:].reshape(1, -1)

    # Compute cosine similarity between target movie and all others
    cos_sim = cosine_similarity(gen_data_matrix, gen_data_select_movie)

    # Create dataframe of movieId and cosine similarity, sort by similarity
    movie_similarity = pd.DataFrame({'movieId': gen_data['movieId'],
                                     'cos_sim': np.squeeze(cos_sim)})
    movie_similarity = movie_similarity.sort_values(by=['cos_sim'], ascending=False)

    # Return top matches
    return movie_similarity['movieId'][1:num_movies+1]