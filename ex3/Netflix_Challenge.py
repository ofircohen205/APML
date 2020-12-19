from sklearn import datasets, metrics
from mpl_toolkits.mplot3d import Axes3D
from utils import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE, LocallyLinearEmbedding, locally_linear_embedding
from sklearn.cluster import SpectralClustering, Birch, AgglomerativeClustering


class Netflix_Challenge:
    """
    Netflix challenge class for manifold learning and clustering
    """
    def __init__(self, remove_empty_cols=False, use_genres=False, most_viewed_movies_limit=0, movies_per_user_limit=500):
        self.netflix_dataset, self.movies_genres_df = create_initial_data(remove_empty_cols, use_genres)
        self.genres = list(filter(lambda col: col not in ['year_of_release', 'title'], self.movies_genres_df.columns))
        self.df_genres = self.movies_genres_df[self.genres]
        self.movies_info = self.movies_genres_df.filter(['year_of_release', 'title'])
        self.most_viewed_movies_limit = most_viewed_movies_limit
        self.movies_per_user_limit = movies_per_user_limit
        self.most_viewed_genres_limit = 6
        self.movies_by_genres = self.df_genres.sum(axis=0, skipna=True).sort_values(ascending=False)
        self.most_viewed_genres = self.movies_by_genres[:self.most_viewed_genres_limit].index.tolist()
        self.genres = get_ids_by_genres(self.movies_genres_df.shape[0])
    # End init

    def __inspect__movies__(self):
        plt.xticks(rotation=45)
        plt.bar(self.movies_by_genres.index, self.movies_by_genres)
        plt.show()
    # End function

    def __inspect__rating__(self):
        non_zeros = self.netflix_dataset > 0
        range = (0, 3000)
        movies_by_users = np.squeeze(np.asarray(self.get_movies_by_users_histogram(non_zeros)))
        users_by_movies = np.squeeze(np.asarray(self.get_movies_by_users_histogram(non_zeros)))
        plt.hist(bins=1000, x=movies_by_users, range=range)
        plt.hist(bins=1000, x=users_by_movies, range=range)
        plt.show()
    # End function

    def get_movies_by_users_histogram(self, matrix):
        return matrix.sum(axis=1)
    # End function

    def get_users_by_movies_histogram(self, matrix):
        return matrix.sum(axis=0)
    # End function

    def __get_average_ratings__(self, non_zeros):
        total = self.netflix_dataset.sum(axis=1)
        avg_rating = pd.DataFrame((total / self.get_users_by_movies_histogram(non_zeros)))
        avg_rating.index = avg_rating.index + 1
        self.movies_info['avg_rating'] = (total / self.get_movies_by_users_histogram(non_zeros))
    # End function

    def __reduce__dataset__(self, interval):
        print("Start reducing dataset")
        movies_in_interval_indexes = self.movies_genres_df.loc[self.movies_genres_df['year_of_release'].isin(interval)].index.tolist()
        reduced_dataset = self.netflix_dataset[np.where(movies_in_interval_indexes)]
        self.most_viewed_movies = np.squeeze(np.asarray(np.squeeze((reduced_dataset > 0).sum(axis=1)) > self.most_viewed_movies_limit))
        reduced_dataset = self.netflix_dataset[np.where(self.most_viewed_movies)]
        self.movies_per_user = np.squeeze(np.asarray(np.squeeze((reduced_dataset > 0).sum(axis=0)) < self.movies_per_user_limit))
        choose_sub_samples = np.random.choice([True, False], size=reduced_dataset.shape[1], p=[0.1, 0.9])
        sub_sample = np.where(choose_sub_samples)[0]
        print("End reducing dataset")
        return reduced_dataset[:, sub_sample]
    # End function

    def __execute_clustering__(self, technique, interval):
        manifold_data = self.__execute_manifold__(technique, interval)
        self.__save_object__()
        return Birch(n_clusters=self.most_viewed_genres_limit).fit_predict(manifold_data), manifold_data
        # return AgglomerativeClustering(n_clusters=self.most_viewed_genres_limit).fit_predict(manifold_data)
        # return SpectralClustering(n_clusters=self.most_viewed_genres_limit, eigen_solver='arpack', n_jobs=-1).fit_predict(manifold_data)
    # End function

    def __execute_manifold__(self, technique, interval):
        self.__execute_truncated_SVD__(interval)
        print("End TruncatedSVD from scratch")
        if technique == 'LLE':
            return self.__execute_lle__()
        elif technique == 'TSNE':
            return self.__execute_tsne__()
    # End function

    def __execute_truncated_SVD__(self, interval, n_components=1000):
        reduced_data = self.__reduce__dataset__(interval)
        self.limited_genres = self.genres[(np.where(self.most_viewed_movies)[0])]
        print("Reduced dataset shape is: {}".format(reduced_data.shape))
        print("Start TruncatedSVD from scratch")
        self.reduced_data = TruncatedSVD(n_components=n_components, n_iter=10, random_state=42).fit_transform(reduced_data)
    # End function

    def __execute_lle__(self):
        print("Start LLE from scratch")
        # lle = LocallyLinearEmbedding(n_components=3, n_neighbors=12, n_jobs=-1).fit_transform(self.reduced_data)
        lle, _ = locally_linear_embedding(self.reduced_data, n_neighbors=self.most_viewed_genres_limit, n_components=3, n_jobs=-1)
        print("End LLE from scratch")
        return lle
    # End function

    def __execute_tsne__(self):
        print("Start TSNE from scratch")
        tsne = TSNE(n_components=3, n_jobs=-1).fit_transform(self.reduced_data)
        print("End TSNE from scratch")
        return tsne
    # End function

    def __plot_scatter__(self, manifold_dataset, label, scatter_type):
        if scatter_type == '3D':
            scatter_plot_3d(manifold_dataset, self.limited_genres, self.most_viewed_genres, label)
        else:
            scatter_plot_2d(manifold_dataset, self.limited_genres, self.most_viewed_genres, label)
    # End function

    def __plot_clustering__(self, clusters_labels, dataset):
        print("Start plotting Clustering")
        # Building the label to colour mapping
        colours = {0: 'r', 1: 'g', 2: 'b', 3: 'y', 4: 'c', 5: 'm'}
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111)
        color_vec = [colours[label] for label in clusters_labels]

        all = []
        most_viewed_movies = []
        for i, genre in enumerate(list(set(self.limited_genres))):
            if genre is None:
                continue
            idx = np.where(self.limited_genres == genre)[0]
            a = ax.scatter(dataset[idx, 0], dataset[idx, 1], label=genre, marker='.', c=get_all_colors()[genre])
            if genre in self.most_viewed_genres:
                most_viewed_movies.append(a)
            all.append(a)

        ax.legend(tuple(all), tuple(get_all_colors().keys()))

        plt.figure(figsize=(9, 9))
        plt.scatter(dataset[:, 0], dataset[:, 1], c=color_vec)
        plt.legend(tuple(most_viewed_movies), tuple(self.most_viewed_genres))
        plt.show()
        print("End plotting Clustering")
    # End function

    def __str__(self):
        return "Netflix_Challenge.pkl"

    def __save_object__(self):
        joblib.dump(self, self.__str__())
