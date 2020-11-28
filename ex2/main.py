# Name: Ofir Cohen
# ID: 312255847
# Date: 25/11/2020
from Image_Denoising import *
from utils import *


def mvn(test_dataset, X_train, X_test, time):
    mvn_model = learn_MVN(X_train)
    mvn_test_log_likelihood = np.abs(MVN_log_likelihood(X_test, mvn_model))
    print("Test MVN Log Likelihood: {}".format(mvn_test_log_likelihood))
    print("-------------------------------------------------")
    save_model(mvn_model, './output/mvn/mvn_model_{}.pkl'.format(time))
    img_dict_to_mse = test_denoise(mvn_model, test_dataset, MVN_Denoise)
    print(img_dict_to_mse)
# End function


def gsm(test_dataset, X_train, X_test, time):
    k_mixtures = 10
    gsm_model = learn_GSM(X_train, k_mixtures)
    gsm_test_log_likelihood = np.abs(GSM_log_likelihood(X_test, gsm_model))
    print("Test GSM Log Likelihood: {}".format(gsm_test_log_likelihood))
    print("-------------------------------------------------")
    save_model(gsm_model, './output/gsm/gsm_model_{}.pkl'.format(time))
    img_dict_to_mse = test_denoise(gsm_model, test_dataset, GSM_Denoise)
    print(img_dict_to_mse)
# End function


def test_denoise(model, dataset, denoise_func):
    img_dict_to_mse = {}
    for idx, image in enumerate(grayscale_and_standardize(dataset), 0):
        print("Start with image: {}".format(idx+1))
        noisy_mses, denoised_mses = test_denoising(image, model, denoise_func)
        img_dict_to_mse['noisy_' + str(idx)] = noisy_mses
        img_dict_to_mse['denoised_' + str(idx)] = denoised_mses
        print("-------------------------------------------------")
    return img_dict_to_mse
# End function


def main():
    time = current_time()
    create_dirs()
    train_dataset = load_dataset('./train_images.pickle')
    test_dataset = load_dataset('./test_images.pickle')
    train_patches = sample_patches(train_dataset)
    test_patches = sample_patches(test_dataset)

    # MVN
    # mvn(test_dataset, train_patches, test_patches, time)

    # GSM
    gsm(test_dataset, train_patches, test_patches, time)
# End function


if __name__ == '__main__':
    main()
