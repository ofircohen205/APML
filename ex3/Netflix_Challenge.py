from pydiffmap import diffusion_map, kernel
from sklearn import datasets, metrics
from mpl_toolkits.mplot3d import Axes3D
import pydiffmap
from sklearn.preprocessing import normalize
from utils import *
import matplotlib.pyplot as plt
from sklearn.decomposition import SparsePCA, PCA


class Netflix_Challenge:

    def __init__(self, remove_empty_cols=False, use_genres=False):
        self.netflix_dataset, self.movies_genres_df = create_initial_data(remove_empty_cols, use_genres)
        self.genres = list(filter(lambda col: col not in ['year_of_release', 'title'], self.movies_genres_df.columns))
        self.df_genres = self.movies_genres_df[self.genres]
        self.movies_info = self.movies_genres_df.filter(['year_of_release', 'title'])
        self.most_viewed_movies_limit = 2000
        self.number_of_movies_per_user_limit = 25
        self.reduced_pca_filename = 'reduced_pca.pkl'
        self.diffusion_maps_filename = 'diffusion_maps.pkl'

    def __inspect__movies__(self):
        movies_by_genres = self.df_genres.sum(axis=0, skipna=True).sort_values(ascending=False)
        plt.xticks(rotation=45)
        plt.bar(movies_by_genres.index, movies_by_genres)
        plt.show()

    def __inspect__rating__(self):
        non_zeros = self.netflix_dataset > 0
        range = (0, 3000)
        movies_by_users = np.squeeze(np.asarray(self.get_movies_by_users_histogram(non_zeros)))
        users_by_movies = np.squeeze(np.asarray(self.get_movies_by_users_histogram(non_zeros)))
        plt.hist(bins=1000, x=movies_by_users, range=range)
        plt.hist(bins=1000, x=users_by_movies, range=range)
        plt.show()

    def get_movies_by_users_histogram(self, matrix):
        return matrix.sum(axis=1)

    def get_users_by_movies_histogram(self, matrix):
        return matrix.sum(axis=0)

    def __get_average_ratings__(self, non_zeros):
        total = self.netflix_dataset.sum(axis=1)
        avg_rating = pd.DataFrame((total / self.get_users_by_movies_histogram(non_zeros)))
        avg_rating.index = avg_rating.index + 1
        self.movies_info['avg_rating'] = (total / self.get_movies_by_users_histogram(non_zeros))

    def __reduce__dataset__(self):
        self.most_viewed_movies = np.squeeze(np.asarray(np.squeeze((self.netflix_dataset > 0).sum(axis=1)) > self.most_viewed_movies_limit))
        reduced = self.netflix_dataset[np.where(self.most_viewed_movies)]
        self.number_of_movies_per_user_limit = np.squeeze(np.asarray(np.squeeze((reduced > 0).sum(axis=0)) < self.number_of_movies_per_user_limit))
        reduced = reduced[:, np.where(self.number_of_movies_per_user_limit)[0]]
        choose_sub_samples = np.random.choice([True, False], size=reduced.shape[1], p=[0.1, 0.9])
        sub_sample = np.where(choose_sub_samples)[0]
        return reduced[:, sub_sample]

    def __execute_pca__(self):
        reduced = self.__reduce__dataset__()
        return PCA(n_components=reduced.shape[0]//2).fit_transform(reduced.todense())

    def __execute_diffusion_maps__(self, load_pca, load_diffusion_maps):
        if load_pca:
            print("Loading reduce pca from disk")
            reduced_pca = joblib.load(self.reduced_pca_filename)
        else:
            print("Start reduce pca from scratch")
            reduced_pca = self.__execute_pca__()
            joblib.dump(reduced_pca, self.reduced_pca_filename)
            print("End reduce pca from scratch")
        if load_diffusion_maps:
            print("Load diffusion maps from disk")
            diffusion_maps = joblib.load(self.diffusion_maps_filename)
        else:
            print("Start diffusion maps from scratch")
            diffusion_maps = pydiffmap.diffusion_map.DiffusionMap(pydiffmap.kernel.Kernel(), n_evecs=self.genres.__len__()).fit_transform(reduced_pca)
            joblib.dump(diffusion_maps, self.diffusion_maps_filename)
            print("End diffusion maps from scratch")

        labels = list(self.movies_genres_df.iloc[np.where(self.most_viewed_movies)[0]['title']])
        plot_with_text(diffusion_maps, labels, 1000)

    def __execute_tsne(self):
        pass
