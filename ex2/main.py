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
    for idx, image in enumerate(grayscale_and_standardize(train_dataset), 0):
        print("Start with image: {}".format(idx+1))
        test_denoising(image, mvn_model, MVN_Denoise)
        print("-------------------------------------------------")
    for idx, image in enumerate(grayscale_and_standardize(test_dataset), 0):
        print("Start with image: {}".format(idx+1))
        test_denoising(image, mvn_model, MVN_Denoise)
        print("-------------------------------------------------")
# End function


def gsm(train_dataset, test_dataset, X_train, X_test, time):
    k_mixtures = 10
    gsm_model = learn_GSM(X_train, k_mixtures)
    gsm_train_log_likelihood = GSM_log_likelihood(X_train, gsm_model)
    gsm_test_log_likelihood = GSM_log_likelihood(X_test, gsm_model)
    print("Train MVN Log Likelihood: {}".format(gsm_train_log_likelihood))
    print("Test MVN Log Likelihood: {}".format(gsm_test_log_likelihood))
    print("-------------------------------------------------")
    # save_model(gsm_model, './output/gsm/gsm_model_{}.pkl'.format(time))
    # for idx, image in enumerate(grayscale_and_standardize(train_dataset), 0):
    #     print("Start with image: {}".format(idx + 1))
    #     test_denoising(image, gsm_model, GSM_Denoise)
    #     print("-------------------------------------------------")
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
