# MovieRecommendation
This program is a movie recommendation system which implements an item-based, collaborative filtering algorithm. 

***Important Note: System can only be run on python3. 

Steps to running recommender:

1. Run movieRec.py

2. The program will prompt the user to input a film they like, numerical input must be given. 
Each film is given a moveID in the current Dataset. For example, the film Toy Story's movieID is 1.  
In order to enter a specific film, one can take a look at the movies.csv file which contains all the movies in the set and their ID pairing.

3. The program will then prompt user to enter either Cosine or Pearson as a similarity measure. Testing has shown that Pearson's is more accurate when providing movie recommendations. Thus, for better results Pearson's is the measure to pick. 

4. The program will then prompt the user if they would like to perform testing. Testing provides statistical accuracy measurements such as RMSE and MAE, which show how well the similarity measure that was picked fits with the model. Note: By selecting "y," it will take longer for the results to be returned. 

