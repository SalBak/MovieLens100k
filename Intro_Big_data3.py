import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt


#================QUESTION 1 

# Load the ratings data
ratings = pd.read_csv('u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])

# Load the movies data
movies = pd.read_csv('u.item', sep='|', encoding='latin-1', header=None)

# Load the users data
users = pd.read_csv('u.user', sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])


total_users = ratings['user_id'].nunique()
print(total_users)

# Number of movies rated by each user
movies_rated = ratings.groupby('user_id').size()
print('the movies seen by each user are: ', movies_rated)


#Plot movies rated by users
plt.figure(figsize=(10, 6))
movies_rated.plot(kind='hist', bins=30, edgecolor='black')
plt.title('Number of Movies Rated by Each User')
plt.xlabel('Number of Movies')
plt.ylabel('Number of Users')
plt.show()

# Frequency of each rating
rating_counts = ratings['rating'].value_counts().sort_index()

# Plot
plt.figure(figsize=(10, 6))
rating_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Frequency of Each Rating')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()


#================QUESTION 1 part (b)
from scipy.stats import zscore

# Calculate Z-scores
user_z_scores = zscore(movies_rated)

# Identify outliers of user's ratings (Z-score > 3 or Z-score<-3)
outliers = movies_rated[user_z_scores > 3]

print("Outlier users based on number of ratings:")
print(outliers)

# Convert to a wide format 
wide_format = ratings.pivot(index='user_id', columns='movie_id', values='rating')

# Display the pivoted table
print(wide_format)

# Save to a new CSV file if needed
wide_format.to_csv('wide_format_ratings.csv')


#================QUESTION 2

from sklearn.model_selection import train_test_split

#split data set into 80% training set and 20% testing set 
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# Get top-rated movies based on the training set
top_movies = train_data.groupby('movie_id')['rating'].mean().sort_values(ascending=False).head(5).index.tolist()

# Recommend top-rated movies for all users in the test set
top_rated_recommendations = {user: top_movies for user in test_data['user_id'].unique()}

print("Top-rated Recommendations:")
print(top_rated_recommendations)

# Get all unique movies in the training set
unique_movies = train_data['movie_id'].unique()

# Recommend random movies for all users in the test set
random_recommendations = {
    user: np.random.choice(unique_movies, 5, replace=False).tolist() for user in test_data['user_id'].unique()
}

print("Random Recommendations:")
print(random_recommendations, '\n')

from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_mae_rmse (test_data, recommendations):
    actual_ratings= []
    predicted_ratings=[]
    
    for user, recommended_movies in recommendations.items():
        user_test_data = test_data[test_data['user_id'] == user]
        actual = user_test_data.set_index('movie_id')['rating']
        
        for movie in recommended_movies:
            if movie in actual:
                actual_ratings.append(actual[movie])
                predicted_ratings.append(train_data[train_data['movie_id'] == movie]['rating'].mean())


    mae = mean_absolute_error(actual_ratings, predicted_ratings) if actual_ratings else 0 
    rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings)) if actual_ratings else 0
    
    return mae, rmse

#calculate precision, recall, F1
def precision_recall_f1(test_data, recommendations):
    total_hits = 0
    total_recommended = 0
    total_relevant = 0

    for user, recommended_movies in recommendations.items():
        user_test_data = test_data[test_data['user_id'] == user]
        actual = set(user_test_data['movie_id'])

        hits = actual.intersection(recommended_movies)
        total_hits += len(hits)
        total_recommended += len(recommended_movies)
        total_relevant += len(actual)

    # Calculate Precision, Recall, and F1
    precision = total_hits / total_recommended if total_recommended else 0
    recall = total_hits / total_relevant if total_relevant else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

    return precision, recall, f1

#top rated recommendations evaluation
mae_top, rmse_top = calculate_mae_rmse(test_data, top_rated_recommendations)
precision_top, recall_top, f1_top = precision_recall_f1(test_data, top_rated_recommendations)

# Random Recommendations Evaluation
mae_random, rmse_random = calculate_mae_rmse(test_data, random_recommendations)
precision_random, recall_random, f1_random = precision_recall_f1(test_data, random_recommendations)

# Display Results
print("Top-rated Recommendations Evaluation:")
print(f"MAE: {mae_top}, RMSE: {rmse_top}, Precision: {precision_top}, Recall: {recall_top}, F1: {f1_top}")

print("\nRandom Recommendations Evaluation:")
print(f"MAE: {mae_random}, RMSE: {rmse_random}, Precision: {precision_random}, Recall: {recall_random}, F1: {f1_random}")

#================QUESTION 3

# Convert the data into a user-movie interaction matrix

inter_matrix = pd.pivot_table(ratings,index='user_id', columns='movie_id', values='rating',fill_value=0)


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# r is the ratings matrix
# k is the number of most similar users
def findKSimilar (r, k):
    
    # similarUsers is 2-D matrix
    similarUsers=-1*np.ones((nUsers,k))
    similarities=cosine_similarity(r)
    # for each user
    
    # find the indexes of all  users similar to i, by increasing value of similarity
    for i in range(0, nUsers):
        simUsersIdxs= np.argsort(similarities[:,i])
        
        l=0
        #find its most similar users    
        for j in range(simUsersIdxs.size-2, simUsersIdxs.size-1-k-1,-1):
            #print(simUsersIdxs[-k+1:],l)
            similarUsers[i,l]=simUsersIdxs[j]
            l=l+1
            
    return similarUsers, similarities
    
    
def predict(userId, itemId, r,similarUsers,similarities):
    # number of neighbours to consider
    nCols=similarUsers.shape[1]
    
    sum=0.0;
    simSum=0.0;
    for l in range(0,nCols):    
        neighbor=int(similarUsers[userId, l])
        #weighted sum
        sum= sum+ r.iloc[neighbor,itemId]*similarities[neighbor][userId]
        simSum = simSum + similarities[neighbor,userId]
    
    return  sum/simSum

r=inter_matrix

nUsers=len(inter_matrix)
nItems=len(movies)

nRows=inter_matrix.shape[0]
nCols=inter_matrix.shape[1]

#get the similarities of users 
similarUsers, similarities=findKSimilar (r,2)

#userId=10
#itemId=12
#prediction=predict(userId,itemId,r, similarUsers, similarities)
#print ('prediction, real',prediction, r.iloc[userId,itemId])

pred_matrix=np.zeros([nUsers,nItems])
#calclulate the predicted matrix of every user and item
for u in range(0,nUsers):
    for i in range(0,nItems):
        pred_matrix[u,i]=predict(u,i,r,similarUsers,similarities)

#get top 5 values on each row and retrun the itemIds
def get_top_5_columns(row):
    sorted_indices = row.sort_values(ascending=False).index[:5]
    return sorted_indices.tolist()
  
# convert pred_matrix into DataFrame
prediction_table=pd.DataFrame(data=pred_matrix)
# create new colum with top 5 recommended movies ids for each user
prediction_table['top_5_rec_movies']=prediction_table.apply(get_top_5_columns,axis=1)
user_top_5_recom=prediction_table['top_5_rec_movies']
print('top 5 recommended movies :',user_top_5_recom)

# Q3.4.a hide 20% of the inter_matrix values


def random_sample_from_matrix(matrix, sample_size=0.2):
    num_elements = matrix.size
    num_sample = int(num_elements * sample_size)
    indices = np.random.choice(num_elements, num_sample, replace=False)
    rows, cols = np.unravel_index(indices, matrix.shape)
    #sampled_values = matrix[rows, cols]
    coordinates = list(zip(rows, cols))
    return coordinates

#coordinates of 20% hidden values
coord_20=random_sample_from_matrix(inter_matrix.values)

#Mean absolut error 
mae=0
for i in range(0,len(coord_20)):
    #predicted vale
    pred_val=predict(coord_20[i][0],coord_20[i][1],inter_matrix,similarUsers,similarities)
    #actual value
    actual_val=inter_matrix.iloc[coord_20[i]]
    mae=mae+np.abs(pred_val-actual_val)
mae=mae/(nRows*nCols)
print('Mean absolute error of 20% hidden values (MAE)=',mae)

#RMSE value
RMSE= 0 
for i in range(0,len(coord_20)):
    #predicted vale
    pred_val=predict(coord_20[i][0],coord_20[i][1],inter_matrix,similarUsers,similarities)
    #actual value
    actual_val=inter_matrix.iloc[coord_20[i]]
    RMSE=RMSE+((actual_val-pred_val)**2)
RMSE=np.sqrt(RMSE/len(coord_20))
print('RMSE evaluation on 20% hidden values (RMSE)=',RMSE)

#  Precision , Recall , F1 

#compute Precision: If rating>=3 then the product is good, otherwise it is
#considered as to be bad
# precision = tp/(tp+fp)
# recall = tp/(tp+fn)
tp=fn=fp=fn=0
for i in range(0,len(coord_20)):
    pred_val=predict(coord_20[i][0],coord_20[i][1],inter_matrix,similarUsers,similarities)
    actual_val=inter_matrix.iloc[coord_20[i]]
    if pred_val>=3 and actual_val>=3:
        tp=tp + 1
    elif pred_val >=3 and actual_val<3:
        fp=fp+1
    elif pred_val <3 and actual_val >=3:
        fn=fn+1

precision=tp/(tp+fp)
recall = tp/(tp+fn)
# Precision ,Recall
print ('Precision=',precision)
print ('Recall=',recall)
# F1
if precision !=0 and recall !=0:
    f1=2*precision*recall/(precision+recall)
    print ('F1=',f1)

#================QUESTION 4 
#  Q4.1
#Improvements Tune hyperparameters,different user thresholds and additional features
#1st Tue hyperparameters (change the number of neighbors) 
# inverted matrix of users and movie ratings
inter_matrix = pd.pivot_table(ratings,index='user_id', columns='movie_id', values='rating',fill_value=0)

r=inter_matrix
nUsers=len(inter_matrix)
nItems=len(movies)
nRows=inter_matrix.shape[0]
nCols=inter_matrix.shape[1]
similarUsers, similarities=findKSimilar (r,4)
pred_matrix=np.zeros([nUsers,nItems])
#calclulate the predicted matrix of every user and item
for u in range(0,nUsers):
    for i in range(0,nItems):
        pred_matrix[u,i]=predict(u,i,r,similarUsers,similarities)
       
# random 20% hiden cells
coord_20=random_sample_from_matrix(inter_matrix.values)
#Mean absolut error 
mae=0
for i in range(0,len(coord_20)):
    #predicted vale
    pred_val=predict(coord_20[i][0],coord_20[i][1],inter_matrix,similarUsers,similarities)
    #actual value
    actual_val=inter_matrix.iloc[coord_20[i]]
    mae=mae+np.abs(pred_val-actual_val)
mae=mae/(nRows*nCols)
print('Mean absolute error of 20% hidden values for 4 neighbors (MAE)=',mae) # output (MAE)= 0.058623468367870045

# for 3 neighbors we have 
similarUsers, similarities=findKSimilar (r,3)
pred_matrix=np.zeros([nUsers,nItems])
#calclulate the predicted matrix of every user and item
for u in range(0,nUsers):
    for i in range(0,nItems):
        pred_matrix[u,i]=predict(u,i,r,similarUsers,similarities)
coord_20=random_sample_from_matrix(inter_matrix.values)
mae=0
for i in range(0,len(coord_20)):
    #predicted vale
    pred_val=predict(coord_20[i][0],coord_20[i][1],inter_matrix,similarUsers,similarities)
    #actual value
    actual_val=inter_matrix.iloc[coord_20[i]]
    mae=mae+np.abs(pred_val-actual_val)
mae=mae/(nRows*nCols)
print('Mean absolute error of 20% hidden values for 3 neighbors (MAE)=',mae) # output (MAE)= 0.05865808414885675
# we assume that since for neighbor of 4 give us smaller error then with 3, the more 
# neighbors we allaw the model to have the smaller the errors we will get. But it takes more time 

# Q4.2)  Using threshold for users that have more the 40 ratings
inter_matrix = pd.pivot_table(ratings,index='user_id', columns='movie_id', values='rating',fill_value=0)
#get user id that have more the 40 ratings
get_user=[]
for i in range(0,len(inter_matrix)):
    count_r=0
    for j in inter_matrix.iloc[i]:
        if j > 0:
            count_r=count_r + 1 
            if count_r==40:
                get_user.append(i)
                break
#filter the inter_matrix table    
      
def filter_pivot_table_by_index(pivot_table, filter_list):
    filter_set = set(filter_list)
    filtered_pivot_table = pivot_table[pivot_table.index.isin(filter_set)]
    return filtered_pivot_table

q=filter_pivot_table_by_index(inter_matrix, get_user)
get_user.pop()

r=q

nUsers = len(get_user)
nItems = len(movies)  
similarUsers, similarities=findKSimilar (r,2)  

# get prediction matrix of those users and all ratings
pred_matrix=np.zeros([nUsers,nItems])
#calclulate the predicted matrix of every user and item
for u in range(0,nUsers):
    for i in range(0,nItems):
        pred_matrix[u,i]=predict(u,i,r,similarUsers,similarities)

#get top 5 values on each row and retrun the itemIds
def get_top_5_columns(row):
    sorted_indices = row.sort_values(ascending=False).index[:5]
    return sorted_indices.tolist()
  
# convert pred_matrix into DataFrame
prediction_table=pd.DataFrame(data=pred_matrix)
# create new colum with top 5 recommended movies ids for each user
prediction_table['top_5_rec_movies']=prediction_table.apply(get_top_5_columns,axis=1)
user_top_5_recom=prediction_table['top_5_rec_movies']
print('top 5 recommended movies :',user_top_5_recom)

# Q4.3) incorporate additional features like timestamp

inter_matrix = pd.pivot_table(ratings,index='user_id', columns='movie_id', values='rating',fill_value=0)
time_matrix=pd.pivot_table(ratings,index='user_id', columns='movie_id', values='timestamp',fill_value=0)

t=inter_matrix
r=time_matrix

nUsers=len(time_matrix)
nItems=len(movies)

nRows=time_matrix.shape[0]
nCols=time_matrix.shape[1]

similarUsers,similarities = findKSimilar(t,2)
#get the similarities of users based on timestamp
similarUsersTime, similaritiesTime=findKSimilar (r,2)

NewsimilarUsers=0.4*similarUsers+0.6*similarUsersTime
Newsimilarities=0.4*similarities+0.6*similaritiesTime

pred_matrix=np.zeros([nUsers,nItems])
#calclulate the predicted matrix of every user and item
for u in range(0,nUsers):
    for i in range(0,nItems):
        pred_matrix[u,i]=predict(u,i,r,NewsimilarUsers,Newsimilarities)

#get top 5 values on each row and retrun the itemIds
def get_top_5_columns(row):
    sorted_indices = row.sort_values(ascending=False).index[:5]
    return sorted_indices.tolist()
  
# convert pred_matrix into DataFrame
prediction_table=pd.DataFrame(data=pred_matrix)
# create new colum with top 5 recommended movies ids for each user
prediction_table['top_5_rec_movies']=prediction_table.apply(get_top_5_columns,axis=1)
user_top_5_recom=prediction_table['top_5_rec_movies']
print('top 5 recommended movies based on timestamps :',user_top_5_recom)


