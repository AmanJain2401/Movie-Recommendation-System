import numpy
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

#(A) importing the dataset for movie recommend which is 100k of data
column_name = ['user_id','item_id','rating','timestamp']
df = pd.read_csv("C:/Users/AMAN/Desktop/Python programs/Python MC/PythonforDataScience/Movie recommendation system/u.data", sep='\t', names=column_name)
#print(df.head())
#print(df.shape)

movies_title = pd.read_csv("C:/Users/AMAN/Desktop/Python programs/Python MC/PythonforDataScience/Movie recommendation system/u.item", sep='|', header=None, encoding='latin-1')
#print(movies_title)

#we only nee d1st two columns item id and names:
movies_titles = movies_title[[0,1]]
movies_titles.columns = ['item_id', 'title']
#print(movies_titles.head())

df = pd.merge(df, movies_titles, on="item_id")
#print(df.head())

"""
#extract unique users and movies:
print(df['user_id'].nunique())
print(df['item_id'].nunique())
"""

#(B)EXPLORATORY DATA ANALYSIS
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

#Finding average rating of each movie by grouping data accord to title:
#df = df.groupby('title').mean()['rating'].sort_values(ascending = False).head()
#print(df)

#Now avoid movies with only one review by counting no of reviews per movie:
#df = df.groupby('title').count()['rating'].sort_values(ascending=False)
#print(df)

#Lets create a dataframe of ratings:
ratings = pd.DataFrame(df.groupby('title').mean()['rating'])
ratings['num of ratings'] = pd.DataFrame(df.groupby('title').count()['rating'])
#print(ratings.head()) #Gives us avg ratings and number of ratings

#No of ratings with 1 or 2 or less ratings:
ratings = ratings.sort_values(by='rating', ascending=False)
#print(ratings)

#Plotting histogram x-axis: no of ratings, y-axis: times no of ratings appeared:
plt.figure(figsize=(10,6))
plt.hist(ratings['num of ratings'], bins=70)
#plt.show()

#Plotting histogram of ratings x-average rating, y-frequency :
plt.hist(ratings['rating'], bins=70)
#plt.show()

#Join plot between no of rating and average rating x-avg rating, y-no of ratings:

sns.jointplot(x='rating',y='num of ratings',data=ratings, alpha = 0.5)
#plt.show()

#(C) CREATING RECCOMENDATION SYSTEM:
#creating matrix with user id at (rows) and movie title (columns) on another axis:
#and each cell containing rating of the particular user(row) to particular movie(column):

moviemat = df.pivot_table(index='user_id', columns='title', values='rating')
#print(moviemat)
starwars_user_ratings = moviemat['Star Wars (1977)']
#print(starwars_user_ratings)

#Correalating (how correleated star wars movie is with other movies based on rating):
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
corr_starwars = pd.DataFrame(similar_to_starwars, columns=['correlation'])
#print(corr_starwars)

#Remove NAN Values in correleations:
corr_starwars.dropna(inplace=True) #using inplace we use it to return same values
#print(corr_starwars.head())

#x = corr_starwars.sort_values('correlation', ascending=False).head(10)

#Filtering as using only those movies with no. of reviews > 100 by using a threshold:
corr_starwars = corr_starwars.join(ratings['num of ratings'])
#print(corr_starwars)
corr_starwars = corr_starwars[corr_starwars['num of ratings']>100].sort_values('correlation', ascending=False)
#print(corr_starwars)

##PREDICTIOM FUNCTION
def predict_movies(movie_name):
    movie_user_rating = moviemat[movie_name]
    similar_to_movie = moviemat.corrwith(movie_user_rating)
    corr_movie = pd.DataFrame(similar_to_movie, columns=['correlation'])
    # Remove NAN Values in correleations:
    corr_movie.dropna(inplace=True)  # using inplace we use it to return same values
    corr_movie = corr_movie.join(ratings['num of ratings'])
    predictions = corr_movie[corr_movie['num of ratings']>100].sort_values('correlation', ascending=False)
    return predictions

predictions = predict_movies('True Lies (1994)')
print(predictions)