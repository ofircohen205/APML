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
        self.movies_info_file_name = 'movies_info.pkl'
        self.path = './archive/'
        self.files_path = './dataset/'
        self.csv_prefix = 'mat_of_movies_and_users_'

    def create_movies_info_dataset(self, file_name='movie_titles.csv'):
        if not os.path.exists(self.movies_info_file_name):
            with open(os.path.join(self.path, file_name), 'rb') as f:
                df_of_movies_info = pd.read_csv(f, error_bad_lines=False, encoding='latin-1', index_col=0,
                                                names=['year_of_release', 'title'])
                if self._use_genres:
                    df_of_movies_genres = self.get_genres_of_movies()
                    df_of_movies_info = df_of_movies_info.join(df_of_movies_genres)
                    df_of_movies_info.fillna(0, inplace=True)

            self.save_created_files(df_of_movies_info, self.movies_info_file_name)
            return load_dataset(self.files_path + self.movies_info_file_name)

        return load_dataset(self.files_path + self.movies_info_file_name)
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

    def create_netflix_dataset(self):
        """
        :param path: path for the archive folder
        :return:
        """
        if not os.path.exists(self.netflix_matrix_file_name):
            print("Started processing the data from scratch")
            # this matrix has movies indices as rows and user ids as columns, and inside it there's the rating
            dataframes_list = []
            for idx, filename in enumerate(os.listdir(self.path), 0):
                if 'combined_data' in filename:
                    self.parse_single_ratings_file(os.path.join(self.path, filename), idx+1)

            mat_of_movies_and_users_df = self.combine_dataset_files()
            return mat_of_movies_and_users_df
        else:
            return load_dataset('./dataset/{}'.format(self.netflix_matrix_file_name))
    # End function

    def parse_single_ratings_file(self, fname, number):
        """
        This function handles a single ratings' file - parses the file and saves its data in the sparse matrix
        :param fname: path for the file we read from
        :param number: number of the file we are reading
        :return:
        """
        print("Start saving txt file number {} as csv file".format(number))
        # mat_of_movies_and_users = scipy.sparse.lil_matrix((17_770, 2_649_429))
        rows_list = []
        movie_id = 1
        with open(fname, 'r') as f:
            for line in f:
                if ',' in line:
                    customer_id, rating, date = line.split(',')
                    date = parser.parse(date)
                    rating = int(rating)
                    customer_id = int(customer_id)
                    rows_list.append({
                        'movie_id': movie_id - 1,
                        'customer_id': customer_id,
                        'rating': rating})
                else:
                    if line.__contains__(':'):
                        movie_id = int(line.split(':')[0])
                        if rows_list.__len__() > 0:
                            inner_number = str(number) + str(movie_id)
                            pd_to_save_name = self.files_path + self.csv_prefix + '{}.csv'.format(inner_number)
                            mat_of_movies_and_users = pd.DataFrame(rows_list)
                            mat_of_movies_and_users.to_csv(pd_to_save_name)
                            rows_list = []
        print("End saving txt file number {} as csv file".format(number))
    # End function

    def combine_dataset_files(self):
        print("Start combining dataset files")
        dataframes_list = []
        mat_of_movies_and_users_df = pd.DataFrame()
        for idx, filename in enumerate(os.listdir(self.files_path), 0):
            if "mat_of_movies_and_users_" in filename:
                csv_file = self.files_path + filename
                df = pd.read_csv(csv_file)
                df.drop(df.columns[[0]], axis=1)
                dataframes_list.append(df)
                mat_of_movies_and_users_df = pd.concat(dataframes_list)
        self.save_created_files(mat_of_movies_and_users_df, self.netflix_matrix_file_name)
        print("End combining dataset files")
        return load_dataset(self.files_path + self.netflix_matrix_file_name)

    def remove_empty_cols_of_sparse_matrix(self, mat_of_movies_and_users):
        """
        This function receives the original matrix of movies and users' ratings and removes empty columns
        (ids of users who have no ratings for any movie)
        :param mat_of_movies_and_users:
        :return:
        """
        print("Started removing empty cols of matrix")
        indices = np.nonzero(mat_of_movies_and_users)
        columns_non_unique = indices[1]
        unique_columns = sorted(set(columns_non_unique))
        mat_of_movies_and_users = mat_of_movies_and_users.tocsc()[:, unique_columns]
        print("Finished removing empty cols of matrix")
        return mat_of_movies_and_users
    # End function

    def load_files_from_disk(self):
        """
        This function loads both files from disk if they exist (one file is the ratings matrix and the other is the
        dataframe with the information about the movies)
        :return:
        """
        print("Started loading data from disk")
        mat_of_movies_and_users = scipy.sparse.load_npz(self.netflix_matrix_file_name).tolil()
        df_of_movies_info = joblib.load(self.movies_info_file_name)
        print("Finished loading data from disk")
        return df_of_movies_info, mat_of_movies_and_users
    # End function

    def save_created_files(self, df, name):
        """
        This function saves the files which were created
        :param df: dataframe to save
        :param name: name of the file
        :return:
        """
        try:
            print("Started saving pickle file")
            joblib.dump(df, name)
            print("Finished saving pickle file")
        except Exception as e:
            print("failed to save file")
            print(e)
    # End function
# End class
