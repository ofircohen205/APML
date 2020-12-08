from sklearn import datasets, metrics
from mpl_toolkits.mplot3d import Axes3D
from utils import *
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from sklearn.cluster import SpectralClustering


class Netflix_Challenge:
    """
    Netflix challenge class for manifold learning and clustering
    """
    def __init__(self, remove_empty_cols=False, use_genres=False):
        self.netflix_dataset, self.movies_genres_df = create_initial_data(remove_empty_cols, use_genres)
        self.genres = list(filter(lambda col: col not in ['year_of_release', 'title'], self.movies_genres_df.columns))
        self.df_genres = self.movies_genres_df[self.genres]
        self.movies_info = self.movies_genres_df.filter(['year_of_release', 'title'])
        self.most_viewed_movies_limit = 0
        self.most_viewed_genres = ['Drama', 'Comedy', 'Romance', 'Thriller', 'Action', 'Crime']
    # End init

    def __inspect__movies__(self):
        movies_by_genres = self.df_genres.sum(axis=0, skipna=True).sort_values(ascending=False)
        plt.xticks(rotation=45)
        plt.bar(movies_by_genres.index, movies_by_genres)
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
        choose_sub_samples = np.random.choice([True, False], size=reduced_dataset.shape[1], p=[0.1, 0.9])
        sub_sample = np.where(choose_sub_samples)[0]
        print("End reducing dataset")
        return reduced_dataset[:, sub_sample]
    # End function

    def __execute_clustering__(self, technique, interval):
        manifold_data = self.__execute_manifold__(technique, interval)
        self.__plot_scatter__(manifold_data, technique, '2D')
        return SpectralClustering(n_clusters=self.most_viewed_genres.__len__(),
                           affinity='rbf', n_jobs=-1).fit_predict(manifold_data)
    # End function

    def __execute_manifold__(self, technique, interval):
        reduced_data = self.__execute_truncated_SVD__(interval)
        print("End TruncatedSVD from scratch")
        if technique == 'LLE':
            return self.__execute_lle__(reduced_data)
        elif technique == 'TSNE':
            return self.__execute_tsne__(reduced_data)
    # End function

    def __execute_truncated_SVD__(self, interval, n_components=1000):
        reduced_data = self.__reduce__dataset__(interval)
        print("Reduced dataset shape is: {}".format(reduced_data.shape))
        print("Start TruncatedSVD from scratch")
        return TruncatedSVD(n_components=n_components).fit_transform(reduced_data)
    # End function

    def __execute_lle__(self, reduced_data):
        print("Start LLE from scratch")
        lle = LocallyLinearEmbedding(n_components=3, n_neighbors=12, n_jobs=-1).fit_transform(reduced_data)
        print("End LLE from scratch")
        return lle
    # End function

    def __execute_tsne__(self, reduced_data):
        print("Start TSNE from scratch")
        tsne = TSNE(n_components=3, n_jobs=-1).fit_transform(reduced_data)
        print("End TSNE from scratch")
        return tsne
    # End function

    def __plot_scatter__(self, manifold_dataset, label, scatter_type):
        genres = get_ids_by_genres(self.movies_genres_df.shape[0])
        limited_genres = genres[(np.where(self.most_viewed_movies)[0])]
        if scatter_type == '3D':
            scatter_plot_3d(manifold_dataset, limited_genres, label)
        else:
            scatter_plot_2d(manifold_dataset, limited_genres, label)
    # End function

    def __plot_clustering__(self, clusters_labels):
        print("Start plotting Clustering")
        limited_genres = get_ids_by_genres(self.movies_genres_df.shape[0])[(np.where(self.most_viewed_movies)[0])]
        # Building the label to colour mapping
        colours = {0: 'b', 1: 'y', 2: 'r', 3: 'g', 4: 'c', 5: 'm'}
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111)
        cvec = [colours[label] for label in clusters_labels]

        # Plotting the clustered scatter plot
        # b = ax.scatter(dataset[:, 0], dataset[:, 1], color='b')
        # y = ax.scatter(dataset[:, 0], dataset[:, 1], color='y')
        # r = ax.scatter(dataset[:, 0], dataset[:, 1], color='r')
        # g = ax.scatter(dataset[:, 0], dataset[:, 1], color='g')
        # c = ax.scatter(dataset[:, 0], dataset[:, 1], color='c')
        # m = ax.scatter(dataset[:, 0], dataset[:, 1], color='m')

        all_colors = {
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

        most_viewed_movies = []
        most_viewed_movies_genres = ['Drama', 'Comedy', 'Romance', 'Thriller', 'Action', 'Crime']
        for i, genre in enumerate(limited_genres):
            idx = np.where(limited_genres == genre)[0]
            a = ax.scatter(dataset[idx, 0], dataset[idx, 1], label=genre, marker='.', c=all_colors[genre])
            if genre in most_viewed_movies_genres:
                most_viewed_movies.append(a)

        plt.figure(figsize=(9, 9))
        plt.scatter(dataset[:, 0], dataset[:, 1], c=cvec)
        plt.legend(tuple(most_viewed_movies), ('Drama', 'Comedy', 'Romance', 'Thriller', 'Action', 'Crime'))
        plt.show()
        print("End plotting Clustering")
    # End function
