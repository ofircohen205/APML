# APML: Exercise 1
## Author: Ofir Cohen | ID: 312255847

### How to execute the program
open utils.py and in init function set the 'train_dataset_path' to the path where the dataset is, also for 'dev_dataset_path'
open bash and execute the command: python main.py

### Files:
1. models.py - The learning model
2. dataset.py - Loads the dataset
3. evaluator.py - Evaluates a given model on dataset
4. trainer.py - Trains a given model
5. adversarial.py - Generates FSGM attack on given model
6. utils.py - Utilities functions
7. main.py - Given the learning model and dataset, we first wish to evaluate the given model and train it in order to decrease the loss if needed.

### main.py:
First of all, init function *MUST* executed all the time!
it is loading the train and dev datasets from disk and creates the hyper-parameters for the learning process
Second, the main is divided into few sub functions:
* Data inspection - inpsect our given dataset
* Data Loader creation - creates DataLoader for train and dev datasets
* Full Process - Evaluate, train and improve given model
* Evaluation - evaluate pretrain model
* Training Loop - evaluate and execute a training loop on pretrain model
* Playing with Learning Rate - train model on various number of learning rates
* Adversarial Example - in order to execute, please execute train_and_eval first or execute the whole process