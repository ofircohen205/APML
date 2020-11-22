# Name: Ofir Cohen
# ID: 312255847
# Date: 25/11/2020
from utils import *
from Image_Denoising import *
from utils import *


def mvn(train_dataset, test_dataset, X_train, X_test, time):
    mvn_model = learn_MVN(X_train)
    mvn_train_log_likelihood = MVN_log_likelihood(X_train, mvn_model)
    mvn_test_log_likelihood = MVN_log_likelihood(X_test, mvn_model)
    print("Train MVN Log Likelihood: {}".format(mvn_train_log_likelihood))
    print("Test MVN Log Likelihood: {}".format(mvn_test_log_likelihood))
    print("-------------------------------------------------")
    save_model(mvn_model, './output/mvn/mvn_model_{}.pkl'.format(time))
    test_denoise(mvn_model, train_dataset, MVN_Denoise)
    test_denoise(mvn_model, test_dataset, MVN_Denoise)
# End function


def gsm(train_dataset, test_dataset, X_train, X_test, time):
    k_mixtures = 5
    gsm_model = learn_GSM(X_train, k_mixtures)
    gsm_train_log_likelihood = GSM_log_likelihood(X_train, gsm_model)
    gsm_test_log_likelihood = GSM_log_likelihood(X_test, gsm_model)
    print("Train MVN Log Likelihood: {}".format(gsm_train_log_likelihood))
    print("Test MVN Log Likelihood: {}".format(gsm_test_log_likelihood))
    print("-------------------------------------------------")
    save_model(gsm_model, './output/gsm/gsm_model_{}.pkl'.format(time))
    test_denoise(gsm_model, train_dataset, GSM_Denoise)
    test_denoise(gsm_model, test_dataset, GSM_Denoise)
# End function


def test_denoise(model, dataset, denoise_func):
    for idx, image in enumerate(grayscale_and_standardize(dataset), 0):
        print("Start with image: {}".format(idx+1))
        test_denoising(image, model, denoise_func)
        print("-------------------------------------------------")
# End function


def main():
    time = current_time()
    create_dirs()
    train_dataset = load_dataset('./train_images.pickle')
    test_dataset = load_dataset('./test_images.pickle')
    train_patches = sample_patches(train_dataset)
    test_patches = sample_patches(test_dataset)

    # MVN
    # mvn(train_dataset, test_dataset, train_patches, test_patches, time)

    # GSM
    gsm(train_dataset, test_dataset, train_patches, test_patches, time)
# End function


if __name__ == '__main__':
    main()
