## Data Analysis is design to perform exploratory data analysis & 
# visualization on a dataset of tweets.## 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import calendar as cal
import datetime as dt
import os

os.makedirs('output', exist_ok=True)

class DataAnalysis:
  def __init__(self, data):
    self.data = data
    self.tweet_count = None

  def count_tweets(self):
    self.tweet_count = self.data['user'].value_counts().reset_index(name='count').rename(columns = {'index': 'user'})
    # values for the bins
    bins = [0, 5, 10, 15, 20, float('inf')]
    labels = ['<5', '6–10', '11–15', '16–20', '20+']
    self.tweet_count['tweet_count_bin'] = pd.cut(self.tweet_count['count'], bins=bins, labels=labels)


  def date_analysis(self):
    print(f"Unique Year: {self.data['year'].unique()}")
    unique_month = [ cal.month_name[i] for i in self.data['month'].unique()]
    print(f"Unique Month: {unique_month}")
    print(f"Unique Day: {np.sort(self.data['day'].unique())}")
    print(f"Unique Users: {len(self.data['user'].unique())}")
    print(f"Unique flag: {self.data['target'].unique()}")

  # Plot the number of tweets by month
  def plot_by_month(self):
    plt.figure(figsize=(10, 6))
    sns.histplot(self.data['month'], kde=True, bins = 12, color= 'skyblue')
    plt.xlabel('Month')
    plt.ylabel('Tweet Count')
    plt.title('Distribution of Tweets by Month')
    plt.savefig('output/plot_by_month.png')
    plt.close()

  #Plot the number of tweets by day
  def plot_by_day(self):
    plt.figure(figsize=(10, 6))
    sns.histplot(self.data['day'], kde=True, bins = 31, color = 'skyblue')
    plt.xlabel('Day')
    plt.ylabel('Tweet Count')
    plt.title('Distribution of Tweets by Day')
    plt.tight_layout()
    plt.savefig('output/plot_by_day.png')
    plt.close()

  # PLot the number of tweets by user
  def plot_tweets_per_user(self):
    self.count_tweets()
    plt.figure(figsize=(10, 6))
    sns.countplot(data=self.tweet_count, x='tweet_count_bin', hue = "tweet_count_bin", palette="Set2", legend=False)
    plt.title('Number of Tweets per User')
    plt.xlabel('Tweet Count Bin')
    plt.ylabel('Number of Users')
    plt.tight_layout()
    plt.savefig('output/plot_tweets_per_user.png')
    plt.close()

  #Plot 20 users with the most tweets
  def plot_by_user(self):
    user_count = self.data.groupby('user').size().reset_index(name='count')
    user_count = user_count.sort_values(by='count', ascending=False).head(20)

    # print(user_count)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=user_count, x='user', y='count', hue = 'user', palette = "Set2")
    plt.title('Top 20 Users with the Most Tweets')
    plt.xlabel('Users')
    plt.ylabel('Tweet Count')
    plt.xticks(rotation = 45, ha = 'right')
    plt.tight_layout()
    plt.savefig('output/plot_by_user.png')
    plt.close()

  # Plot the total number of positive & Negative tweets
  def plot_by_tweeter(self):
    plt.figure(figsize=(10, 6))
    sns.countplot(data = self.data, x = 'target', hue = 'target', palette = 'pastel')
    plt.title('Tweet Count by Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Tweet Count')
    plt.legend(['0 = Negative', '1 = Positive'])
    plt.tight_layout()
    plt.savefig('output/plot_by_tweeter.png')
    plt.close()

  # Plot the comparison of positive & negative tweets
  def plot_user_target(self):
    self.count_tweets()
    user_count = self.data.groupby(['user', 'target']).size().reset_index(name='count')
    user_count = user_count.merge(self.tweet_count[['user', 'tweet_count_bin']], on='user', how='left')
    user_count_bin = user_count.groupby(['tweet_count_bin','target'], observed = True)['count'].sum().reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=user_count_bin, x='tweet_count_bin', y='count', hue = 'target', palette = 'coolwarm')
    plt.title('Sentiment vs Number of Tweets per User Group')
    plt.xlabel('Tweet Count Bin')
    plt.ylabel('Tweet Count')
    plt.legend(['0 = Negative', '1 = Positive'])
    plt.tight_layout()
    plt.savefig('output/plot_user_target.png')
    plt.close()

  def plot_by_time(self):
    data_time = self.data.copy()
    data_time['hour'] = data_time['time'].apply(lambda x: x.hour if pd.notnull(x) else np.nan)

    bins = [0, 6, 12, 18, 24]
    labels = ['Night (0–6)', 'Morning (6–12)', 'Afternoon (12–18)', 'Evening (18–24)']

    data_time['time_bin'] = pd.cut(data_time['hour'], bins=bins, labels=labels)

    time_counts = data_time.groupby(['time_bin', 'target'], observed = True).size().reset_index(name='count')

    plt.figure(figsize=(10, 6))
    sns.barplot(data = time_counts, x = 'time_bin', y ='count', hue = 'target', palette = 'coolwarm')
    plt.title('Tweet Sentiment by Time of Day')
    plt.xlabel('Time of Day')
    plt.ylabel('Tweet Count')
    plt.tight_layout()
    plt.savefig('output/plot_by_time.png')
    plt.close()

  def run_all(self):
    self.date_analysis()
    self.plot_by_month()
    self.plot_by_day()
    self.plot_tweets_per_user()
    self.plot_by_user()
    self.plot_by_tweeter()
    self.plot_user_target()
    self.plot_by_time()
