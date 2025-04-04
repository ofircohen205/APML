import matplotlib.colors as mcolors
import pickle
import matplotlib.pyplot as plt
import os
import zipfile
import io
import pandas as pd
from dateutil import parser
import scipy.sparse
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances


PICKLE_FILE_NAME_NETFLIX_MATRIX = 'netflix_matrix'
PICKLE_FILE_NAME_MOVIES_INFO = 'movies_info.pkl'
PICKLE_FILE_NAME_NETFLIX_MATRIX_INT8 = 'netflix_matrix_int8'
ROW_SIZE = 17_770
COL_SIZE = 2_649_429


def create_dirs():
    if not os.path.exists('./plots/'):
        os.mkdir('./plots/')
    if not os.path.exists('./dataset/'):
        os.mkdir('./dataset/')
# End function


def create_initial_data(remove_empty_cols=False, use_genres=False):
    """
    This function reads all the data in the zip folder named 'archive.zip'.
    The file was downloaded from here: https://www.kaggle.com/netflix-inc/netflix-prize-data
    The folder contains files named 'combined_data_X', where in each text file there's info about ratings of users to
    many different movies. Every time a new movie starts, there's a line with its id and ':', and then after this line
    there are many lines with users' rating to this movie, in this format: "'customer_id, rating, date'", where rating is
    between 1 and 5, and date is in the format 'YYYY-MM-DD'.
    Movies' ids span the range 1 to 17,770 and customers' ids span the range 1 to 2,649,429, but many customer ids
    are not in use. Only 480,189 are in use.
    Except for these files, the 'archive.zip' contains a file named 'movie_titles.csv' with information about the
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
    :param remove_empty_cols:
    :param use_genres
    :return:
    """
    if not os.path.exists(PICKLE_FILE_NAME_NETFLIX_MATRIX_INT8 + '.npz') or not os.path.exists(PICKLE_FILE_NAME_MOVIES_INFO):
        print("Started processing the data from scratch")
        # this matrix has movies indices as rows and user ids as columns, and inside it there's the rating
        mat_of_movies_and_users = scipy.sparse.lil_matrix((ROW_SIZE, COL_SIZE))
        with zipfile.ZipFile('archive.zip', 'r') as z:
            with tqdm(total=17_770, position=0, leave=True) as pbar:
                for filename in z.namelist():
                    if 'combined_data' in filename:
                        with z.open(filename, 'r') as f:
                            f = io.TextIOWrapper(f)
                            parse_single_ratings_file(f, mat_of_movies_and_users, pbar)
                    if 'movie_titles.csv' in filename:
                        with z.open(filename, 'r') as f:
                            df_of_movies_info = pd.read_csv(f, error_bad_lines=False, encoding='latin-1', index_col=0,
                                                            names=['year_of_release', 'title'])
                            if use_genres:
                                df_of_movies_genres = get_genres_of_movies()
                                df_of_movies_info = df_of_movies_info.join(df_of_movies_genres)
                                df_of_movies_info.fillna(0, inplace=True)
            if remove_empty_cols:
                mat_of_movies_and_users = remove_empty_cols_of_sparse_matrix(mat_of_movies_and_users)
            save_created_files(df_of_movies_info, mat_of_movies_and_users)
    else:
        df_of_movies_info, mat_of_movies_and_users = load_files_from_disk()
    return mat_of_movies_and_users, df_of_movies_info
# End function


def load_files_from_disk():
    """
    This function loads both files from disk if they exist (one file is the ratings matrix and the other is the
    dataframe with the information about the movies)
    :return:
    """
    print("Started loading data from disk")
    mat_of_movies_and_users = scipy.sparse.load_npz(PICKLE_FILE_NAME_NETFLIX_MATRIX_INT8 + '.npz').tolil()
    df_of_movies_info = joblib.load(PICKLE_FILE_NAME_MOVIES_INFO)
    print("Finished loading data from disk")
    return df_of_movies_info, mat_of_movies_and_users
# End function


def save_created_files(df_of_movies_info, mat_of_movies_and_users):
    """
    This function saves the files which were created
    :param df_of_movies_info:
    :param mat_of_movies_and_users:
    :return:
    """
    try:
        print("Started saving pickle files")
        scipy.sparse.save_npz(PICKLE_FILE_NAME_NETFLIX_MATRIX_INT8, mat_of_movies_and_users.tocsr(), compressed=True)
        joblib.dump(df_of_movies_info, PICKLE_FILE_NAME_MOVIES_INFO)
        print("Finished saving pickle files")
    except Exception as e:
        print("failed to save files")
        print(e)
# End function


def parse_single_ratings_file(f, mat_of_movies_and_users, pbar):
    """
    This function handles a single ratings' file - parses the file and saves its data in the sparse matrix
    :param f:
    :param mat_of_movies_and_users:
    :param pbar:
    :return:
    """
    for line in f:
        if ',' in line:
            customer_id, rating, date = line.split(',')
            date = parser.parse(date)
            rating = int(rating)
            customer_id = int(customer_id)
            mat_of_movies_and_users[movie_id - 1, customer_id - 1] = rating
        else:
            movie_id = int(line.split(':')[0])
            pbar.update()
# End function


def remove_empty_cols_of_sparse_matrix(mat_of_movies_and_users):
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


def get_genres_of_movies():
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


def get_ids_by_genres(all_movies_size):
    genres_df = pd.read_csv('netflix_genres.csv')
    genres = [None] * all_movies_size
    for id, genre in zip(genres_df['movieId'].to_list(), genres_df['genres'].to_list()):
        movie_genres = genre.split('|')
        if len(movie_genres) == 1:
            genres[id] = movie_genres[0]

    return np.asarray(genres)
# End function


def load_dataset(path):
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset
# End function


def calculate_distances(dataset):
    return euclidean_distances(X=dataset)
# End function


def generate_noised_dataset(num_samples=2000, dim=2, orthogonal_matrix_dim=75):
    data = np.random.uniform(-1, 1, (num_samples, dim))
    matrix = np.random.normal(0, 1, size=(orthogonal_matrix_dim, orthogonal_matrix_dim))
    orthogonal_matrix, _ = np.linalg.qr(matrix)
    return data, orthogonal_matrix
# End function


def get_mds_eig_entities(X):
    N, _ = X.shape
    H = np.eye(N) - np.ones((N, N)) / N
    S = -0.5 * np.dot(np.dot(H, (X ** 2)), H)

    # Diagonalize
    eigvals, eigvecs = np.linalg.eigh(S)

    # sort eigenvalues in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return eigvals, eigvecs
# End function


def get_diffusion_maps_eig_entites(X, sigma):
    X_distances = calculate_distances(X)
    N, _ = X_distances.shape

    # Create kernel similiary matrix
    kernel_similarity = np.exp(-X_distances ** 2 / sigma)

    # Normalize rows of K to form Markov Transition Matrix
    markov_transition_matrix = kernel_similarity / kernel_similarity.sum(axis=1).reshape(kernel_similarity.shape[1], 1)

    # Diagonalize
    eigvals, eigvecs = np.linalg.eigh(markov_transition_matrix)

    # sort eigenvalues in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return eigvals, eigvecs
# End function


###################################################################
############################## PLOTS ##############################
###################################################################
def scree_plot(eig_vals, name, file_name):
    plt.figure()
    xs = np.arange(len(eig_vals)) + 1
    plt.plot(xs, eig_vals, 'ro-', linewidth=2)
    plt.title("Scree Plot")
    plt.xlabel("Matrix size")
    plt.ylabel("Eigenvalue")
    name = "Eigenvalues from {name}".format(name=name)
    plt.legend([name], loc='best')
    plt.savefig(file_name)
    plt.clf()
# End function


def scree_plot_noised(eigvals, size, color, label, x_label, y_label):
    sing_values = np.arange(size) + 1
    plt.plot(sing_values, eigvals, f'{color}o-', linewidth=2, label=label)
    plt.title('Scree Plot')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
# End function


def plot_data(dataset, dataset_reduced, color, file_name):
    # plot the data:
    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')
    ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], c=color, cmap=plt.cm.Spectral)
    ax.set_title("Original data")
    ax = fig.add_subplot(212)
    ax.scatter(dataset_reduced[:, 0], dataset_reduced[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.axis('tight')
    plt.xticks([]), plt.yticks([])
    plt.title('Projected data')
    plt.savefig(file_name)
    plt.clf()
# End function


def scatter_plot_2d(data, genres, most_viewed_genres, label='Scatter'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cmap = create_random_cmap(most_viewed_genres.__len__())
    for i, genre in enumerate(most_viewed_genres):
        idx = np.where(genres == genre)[0]
        ax.scatter(data[idx, 0], data[idx, 1], label=genre, marker='.', cmap=cmap)

    genres_iter = list(set(genres))
    cmap = create_random_cmap(len(genres_iter))
    for i, genre in enumerate(genres_iter):
        idx = np.where(genres == genre)[0]
        ax.scatter(data[idx, 0], data[idx, 1], label=genre, marker='.', cmap=cmap)
    ax.set_title(label)
    plt.legend(loc='best')
    plt.show()
# End function


def scatter_plot_all(data, genres, label):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    genres_iter = list(set(genres))
    cmap = create_random_cmap(len(genres_iter))
    for i, genre in enumerate(genres_iter):
        idx = np.where(genres == genre)[0]
        ax.scatter(data[idx, 0], data[idx, 1], label=genre, marker='.', cmap=cmap)
    ax.set_title(label)
    plt.legend(loc='best')
    plt.show()
# End function


def scatter_plot_3d(data, genres, most_viewed_genres, label='Scatter'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, genre in enumerate(most_viewed_genres):
        idx = np.where(genres == genre)[0]
        ax.scatter(data[idx, 0], data[idx, 1], label=genre, marker='.')

    other = []
    for idx, genre in enumerate(genres):
        if genre not in most_viewed_genres:
            other.append(idx)

    ax.scatter(data[other, 0], data[other, 1], data[other, 2], marker='.', label='other', alpha=0.05)
    ax.set_title(label)
    plt.legend(loc='best')
    plt.show()
# End function


def plot_clustering():
    pass
# End function


def create_random_cmap(length):
    cmap = plt.get_cmap(np.random.choice(["Set1", "Set2", "Set3", "Dark2", "Accent"]))
    cmap.colors = cmap.colors[:length]
    cmap.N = length
    cmap._i_bad = length + 2
    cmap._i_over = length + 1
    cmap._i_under = length
    return cmap
# End function


def get_all_colors():
    return {
            'Drama': 'b', 'Comedy': 'y', 'Romance': 'r', 'Thriller': 'g', 'Action': 'c', 'Crime': 'm',
            'Adventure': mcolors.CSS4_COLORS['slateblue'], 'Documentary': mcolors.CSS4_COLORS['slategray'],
            'Horror': mcolors.CSS4_COLORS['gold'], 'Sci-Fi': mcolors.CSS4_COLORS['maroon'],
            'Family': mcolors.CSS4_COLORS['salmon'],
            'Mystery': mcolors.CSS4_COLORS['lightgreen'], 'Fantasy': mcolors.CSS4_COLORS['sienna'],
            'Music': mcolors.CSS4_COLORS['teal'],
            'Animation': mcolors.CSS4_COLORS['pink'], 'Biography': mcolors.CSS4_COLORS['springgreen'],
            'War': mcolors.CSS4_COLORS['skyblue'], 'History': mcolors.CSS4_COLORS['tomato'],
            'Musical': mcolors.CSS4_COLORS['linen'],
            'Sport': mcolors.CSS4_COLORS['olive'], 'Western': mcolors.CSS4_COLORS['crimson'],
            'Short': mcolors.CSS4_COLORS['azure'], 'Talk-Show': mcolors.CSS4_COLORS['burlywood'],
            'Film-Noir': mcolors.CSS4_COLORS['orchid'], 'Reality-TV': mcolors.CSS4_COLORS['indigo'],
            'Game-Show': mcolors.CSS4_COLORS['darkred'],
            'News': mcolors.CSS4_COLORS['darkcyan']
        }
    # End function
