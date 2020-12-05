import pickle, io, os, joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil import parser
import scipy.sparse
from utils import *


class NetflixPreprocessing:
    """
    This class creates reads all the data from netflix prize in the folder named 'archive'.
    The file was downloaded from here: https://www.kaggle.com/netflix-inc/netflix-prize-data
    The folder contains files named 'combined_data_X', where in each text file there's info about ratings of users to
    many different movies. Every time a new movie starts, there's a line with its id and ':', and then after this line
    there are many lines with users' rating to this movie, in this format: "'customer_id, rating, date'", where rating is
    between 1 and 5, and date is in the format 'YYYY-MM-DD'.
    Movies' ids span the range 1 to 17,770 and customers' ids span the range 1 to 2,649,429, but many customer ids
    are not in use. Only 480,189 are in use.
    Except for these files, the 'archive' folder contains a file named 'movie_titles.csv' with information about the
    movies. Each row is in the format: 'movie_id, year_of_release, title'.
    The function also reads another file with the genres of the movies if the 'use_genres' parameter is set to True
    (needs to be downloaded from 'https://github.com/bmxitalia/netflix-prize-with-genres/blob/master/netflix_genres.csv')
    , and adds it to the given information about the movies (joins dataframes).
    This function iterates through all the files mentioned above and builds a sparse matrix (Scipy lil matrix)
    whose rows represent movies' ids and columns represent users' ids.
    Important note - the indices start from 0, whereas the true ids start from one, so you should take it into
    account when relating to a specific movie or user in the matrix.
    After creating the files, the function saves the information abouts the movies to a pickle file, using joblib,
    and saves the large sparse matrix to '.npz' file.
    The function receives a default argument - whether to remove empty columns or not (user ids which have no use).
    Last note - if the files were already created, the function just reads them from the saved files (pickle and npz).
    """

    def __init__(self, use_genres):
        self._use_genres = use_genres
        self.netflix_matrix_file_name = 'netflix_matrix.npz'
        self.netflix_matrix_file_name_csv = 'netflix_matrix.csv'
        self.netflix_matrix_combined_csv = 'netflix_matrix_combined.csv'
        self.movies_info_file_name = 'movies_info.csv'
        self.path = './archive/'
        self.files_path = './'
        self.dataset_path = './dataset/'
        self.csv_prefix = 'mat_of_movies_and_users_'
    # End init

    def create_movies_info_dataset(self, file_name='movie_titles.csv'):
        if not os.path.exists(self.movies_info_file_name):
            print("Started processing the data from scratch")
            with open(os.path.join(self.path, file_name), 'rb') as f:
                df_of_movies_info = pd.read_csv(f, error_bad_lines=False, encoding='latin-1', index_col=0,
                                                names=['year_of_release', 'title'])
                if self._use_genres:
                    df_of_movies_genres = self.get_genres_of_movies()
                    df_of_movies_info = df_of_movies_info.join(df_of_movies_genres)
                    df_of_movies_info.fillna(0, inplace=True)

            df_of_movies_info.to_csv(self.movies_info_file_name)
            print("Ended processing the data from scratch")
            return pd.read_csv(self.movies_info_file_name)
        else:
            print("Loading the movies info data from directory")
            return pd.read_csv(self.movies_info_file_name)
    # End function

    def get_genres_of_movies(self):
        """
        This function reads the file named 'netflix_genres.csv' which has a mapping between movie id and its genre.
        It prints all unique genres and return a dataframe which is a one-hot encoding - the dataframe contains all
        genres as columns, and 1 if this movie is from this genre, and 0 otherwise (because a movie usually corresponds
        to more than one genre)
        :return:
        """
        df_of_genres = pd.read_csv('netflix_genres.csv')
        all_genres = set()
        for movie_genre in df_of_genres['genres'].to_list():
            all_genres.update(movie_genre.split('|'))
        print("all genres are:")
        print(all_genres)
        print("Number of genres is: ")
        print(len(all_genres))

        df_of_movies_and_all_genres = pd.DataFrame(columns=all_genres)
        for idx, row in df_of_genres.iterrows():
            movie_id = row[0]
            movie_genres = row[1].split('|')
            for movie_genre in movie_genres:
                df_of_movies_and_all_genres.loc[movie_id, movie_genre] = 1
        df_of_movies_and_all_genres.fillna(0, inplace=True)
        return df_of_movies_and_all_genres
    # End function

    def create_movies_and_users_dataset(self):
        """
        :return:
        """
        if not os.path.exists(self.netflix_matrix_file_name_csv):
            print("Started processing the data from scratch")
            # this matrix has movies indices as rows and user ids as columns, and inside it there's the rating
            movies_info_df = self.create_movies_info_dataset()
            for idx, filename in enumerate(os.listdir(self.path), 0):
                if 'combined_data' in filename:
                    self.parse_single_ratings_file(os.path.join(self.path, filename), idx+1, movies_info_df)

            self.combine_dataset_files()
            print("Ended processing the data from scratch")
            return pd.read_csv(self.netflix_matrix_file_name_csv)
        else:
            print("Loading the movies and users data from directory")
            return pd.read_csv(self.netflix_matrix_file_name_csv)
    # End function

    def parse_single_ratings_file(self, fname, number, movies_info_df, skip_number=3):
        """
        This function handles a single ratings' file - parses the file and saves its data in the sparse matrix
        :param fname: path for the file we read from
        :param number: number of the file we are reading
        :param skip_number: skip saving rows every X times
        :return:
        """
        from sklearn.utils import shuffle
        print("Start saving txt file number {} as csv file".format(number))
        rows_list = []
        counter = 0
        with open(fname, 'r') as f:
            for line in f:
                if ',' in line:
                    customer_id, rating, date = line.split(',')
                    rating = int(rating)
                    customer_id = int(customer_id)
                    if counter % skip_number == 0:
                        rows_list.append({
                            'movie_id': movie_id,
                            'customer_id': customer_id,
                            'rating': rating})
                    counter += 1
                else:
                    if line.__contains__(':'):
                        if rows_list.__len__() > 0:
                            inner_number = str(number) + str(movie_id)
                            pd_to_save_name = self.dataset_path + self.csv_prefix + '{}.csv'.format(inner_number)
                            mat_of_movies_and_users_df = pd.DataFrame(rows_list)
                            movie_id_info = movies_info_df.iloc[movie_id-1]
                            for col in movies_info_df.columns:
                                if col == 'Unnamed: 0':
                                    continue
                                mat_of_movies_and_users_df[col] = movie_id_info.loc[col]
                            mat_of_movies_and_users_df.to_csv(pd_to_save_name)
                            rows_list = []
                        movie_id = int(line.split(':')[0])
        inner_number = str(number) + str(movie_id)
        pd_to_save_name = self.dataset_path + self.csv_prefix + '{}.csv'.format(inner_number)
        mat_of_movies_and_users_df = pd.DataFrame(rows_list)
        mat_of_movies_and_users_df.to_csv(pd_to_save_name)
        print("End saving txt file number {} as csv file".format(number))
    # End function

    def combine_dataset_files(self):
        print("Start combining dataset files")
        mat_of_movies_and_users_df = pd.DataFrame()
        csv_list = []
        for idx, filename in enumerate(os.listdir(self.dataset_path), 0):
            if "mat_of_movies_and_users_" in filename:
                csv_file = self.dataset_path + filename
                df = pd.read_csv(csv_file)
                csv_list.append(df)
        mat_of_movies_and_users_df = pd.concat(csv_list)
        mat_of_movies_and_users_df.to_csv(self.netflix_matrix_file_name_csv)
        print("End combining dataset files")
    # End function
# End class
