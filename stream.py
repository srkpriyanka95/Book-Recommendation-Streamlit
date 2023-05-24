import streamlit as st
text='Book Recommendations Based on user query'
s='\U0001F4DA'
st.title(f"{text}{s}")

st.markdown("<hr>", unsafe_allow_html=True)
t="Which Book are you looking for?"
th="\U0001F914"

st.subheader(f'{t}{th}')

col=st.columns(1)
book=col[0].text_input("","Type here...")


import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Load Dataset
books_df=pd.read_csv(r"E:\Almabetter\Capstone Project\br\book-recommendation-data\Copy of Books.csv",low_memory=False)
ratings_df=pd.read_csv(r"E:\Almabetter\Capstone Project\br\book-recommendation-data\Copy of Ratings.csv",low_memory=False)
users_df=pd.read_csv(r"E:\Almabetter\Capstone Project\br\book-recommendation-data\Copy of Users.csv",low_memory=False)
user_rating_df = pd.merge(ratings_df,users_df,on='User-ID',how='left')
data=pd.merge(user_rating_df,books_df,on='ISBN',how='inner')

data.drop(columns={'Image-URL-S','Image-URL-M','Image-URL-L'},inplace=True)
Item_based=data[['ISBN','Book-Title','Book-Author']].copy()
# Creating a dataframe by groping the ISBN by total count of ratings for the books and the mean rating for the book
mean_rating=data.groupby('ISBN').agg({'Book-Title':'count','Book-Rating':'mean'}).reset_index()

# Renaming the columns for clarity of data and rounding off the float values in the mean rating column
mean_rating.rename(columns={'Book-Title':'Rating-Count','Book-Rating':'Mean-Rating'},inplace=True)
mean_rating['Mean-Rating'] = mean_rating['Mean-Rating'].round()

# Merging the mean_rating data frame with the Item based dataframe on ISBN column
Item_based=Item_based.merge(mean_rating,on='ISBN')
# Removing the duplicates from the dataframe
Item_based.drop_duplicates(inplace=True)
# Applying basic preprocessing to the book title in the dataframe
# Entire book title is converted into lower case and any characters other than alphabet and numbers are removed using regular expression
Item_based['altered_title'] = Item_based['Book-Title'].apply(lambda x:x.lower())
Item_based['altered_title'] = Item_based['altered_title'].str.replace('[^a-zA-Z0-9]',' ',regex=True)

# Removing the excess spaces between words in the altered title
Item_based['altered_title'] = Item_based['altered_title'].str.replace('\s+',' ',regex=True)
# Applying TfidfVectorizer to the altered titlte column
vectorizer =TfidfVectorizer()
tfidf=vectorizer.fit_transform(Item_based['altered_title'])
# Creating a function to return books with similar names to the query
def similar_books(book_name:str):
  # The book name entered is converted into lower case and removing the additional characters and additional space using regular expressions
  book_name= re.sub('[^a-zA-Z0-9]',' ',book_name.lower())
  book_name=re.sub('\s+'," ",book_name)

  # Transforming the book name into TFIDF Vectorizer
  book_vector=vectorizer.transform([book_name])

  # Computing the similarities between the book name query vector and the tfid matrix
  similarity= cosine_similarity(book_vector,tfidf).flatten()

  # After getting tje similarities, we are getting the top 10 book index similar to query using numpy arg partition
  similar_book_id = np.flip(np.argpartition(similarity,-10)[-10:])

  # With the book index of the books we are creating a dataframe
  similar_books_df = pd.DataFrame(columns=['Book-Title','Book-Author'])

  # using for loop to iterate over the similar book id list and appending the data to the similar_books_df
  for i in similar_book_id:
    similar_books_df = pd.concat([similar_books_df,pd.DataFrame(Item_based.iloc[i][['Book-Title','Book-Author']]).transpose()])
  similar_books_df.reset_index(drop=True,inplace=True)
  similar_books_df.drop_duplicates(inplace=True)
  
  similar_books_df['Book & Author'] = similar_books_df['Book-Title']+ '   '+'---'+'   '+similar_books_df['Book-Author']
  # Returning the similar book Data Frame
  return similar_books_df
final = similar_books(book)

st.markdown("<hr>", unsafe_allow_html=True)
d="Here are few recommendations for you"
sm='\U0001F60A'
b='\U0001F4D6'
p="\u270D\ufe0f"
st.subheader(f"{d}{sm}{b}{sm}")
for i in range(len(final)):
  st.write(f"{b} {final.iloc[i]['Book-Title']}  --- {p}{final.iloc[i]['Book-Author']} ")

