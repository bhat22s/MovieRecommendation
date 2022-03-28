#Saman Bhat Dec.31 2021
#Item-based collaborative filtering system for movie recommendations. 

import io
from surprise.builtin_datasets import get_dataset_dir
from surprise.model_selection.split import KFold
from surprise.prediction_algorithms import knns
from surprise import Dataset 
from surprise import accuracy
 
 
def read_item_names():
    """Read the u.item file from MovieLens 100-k dataset and return two
    mappings to convert raw ids into movie names
    """
    file_name = get_dataset_dir() + '/ml-100k/ml-100k/u.item'
    rid_to_name = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]
             
    return rid_to_name 

#This function returns the name of the movie based off the movieID 
def convert(id): 
    rid_to_name = read_item_names() 
    return rid_to_name.get(id)

def get_top_n(item, algorithm, testing, n=10):

    #Loads data from built-in dataset 
    data = Dataset.load_builtin('ml-100k')

    #Refines the dataset such that only the relevant portions for the algorithms remain (i.e. userID, movieID, movieTitle, etc.) 
    trainset = data.build_full_trainset() 
    
    #Checks which similarity confguration to use and calls it accordingly 
    if algorithm == "Pearson":
        sim_options = {'name': 'pearson', 'user_based': False}
        algo = knns.KNNBaseline(sim_options=sim_options)
    elif algorithm == "Cosine":
        sim_options = {'name': 'cosine', 'user_based': False}
        algo = knns.KNNBaseline(sim_options=sim_options)
    
    
    # Train the algorithm on the trainset, and predict ratings for the testset
    algo.fit(trainset)

    #Prompts user to submit another movie if movie they gave is not in database 
    while algo.trainset.knows_item(int(item)) != True:
        print("Sorry there is no such movie in out database try again:")
        item = input("What movie do you like: ")
    
    #Returns accuracy metrics if the user wants to perform testing 
    if testing == True:
        testset = trainset.build_testset()
        predictions = algo.test(testset)
    
        accuracy.rmse(predictions, verbose=True) 
        accuracy.mse(predictions, verbose=True)
        accuracy.mae(predictions, verbose=True)
        accuracy.fcp(predictions, verbose=True)
        
    #Converts Raw ID into an into innerID
    inner_id = algo.trainset.to_inner_iid(item)

    #This returns a list of 10 similar movies, basing its descision off of the selected algorithm
    movie_neighbors = algo.get_neighbors(inner_id, k=10)

    #This converts the movie's inner_ID's to their rawIDs  
    movie_neighbors = (algo.trainset.to_raw_iid(inner_id)
                        for inner_id in movie_neighbors)
    
    # Creates a new list with the movie titles, based off the movies rawID's, the first item on the list is the given movie
    newList = [convert(item)]
    for element in movie_neighbors:
        newList.append(convert(element))
    return newList

#This function prints out the recommended movies 
def recommend(item_id, algorithm, testing):
    results = get_top_n(item_id, algorithm, testing)
    print("Recommending 10 products similar to " + results[0] + "...")   
    print("-------")    
    for rec in results[1:]: 
        print("Recommended: " + rec)

#Main function 
def main():
    user_input = input("What movie do you like (input movieID): ")
    print("Select an algorithm: Cosine, Pearson")
    user_algo = input("Select a similarity measure: ")
    testing = input("Would you like to perform testing (y/n): ")
    if testing == "y":
        recommend(user_input, user_algo, True)
    else:
        recommend(user_input, user_algo, False)


if __name__ == "__main__":
    main()

 