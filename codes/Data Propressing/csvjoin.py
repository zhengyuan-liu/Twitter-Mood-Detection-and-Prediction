import pandas as pd
import csv





thisTweets = {}
uid = []
created_at =[]
hashtags = []
retweet_count = []
favorite_count = []
user_name = []
user_friends_count = []
user_followers_count = []
user_location = []
user_tweet_count = []
place= []
geo = []
tweets = []
happ = []

def readCsv(filename):
    thisTweet = pd.read_csv(filename)
    uid.extend( thisTweet['user_id'].values)
    created_at.extend( thisTweet["created_at"].values)
    hashtags.extend( thisTweet["hashtags"].values)
    retweet_count.extend( thisTweet["retweet_count"].values)
    favorite_count.extend( thisTweet["favorite_count"].values)
    user_name.extend( thisTweet["user_name"].values)
    user_friends_count.extend( thisTweet["user_friends_count"].values)
    user_followers_count.extend( thisTweet["user_followers_count"].values)
    user_location.extend( thisTweet["user_location"].values)
    user_tweet_count.extend( thisTweet["user_tweet_count"].values)
    place.extend(thisTweet["place"].values)
    geo.extend( thisTweet["geo"].values)
    tweets.extend( thisTweet["tweets"].values)
    happ.extend(thisTweet["happiness_index"].values)


if __name__ == '__main__':
    readCsv("birthday.csv")
    readCsv("christmas.csv")
    readCsv("hurr.csv")
    readCsv("massshooting.csv")
    readCsv("taxreform.csv")
    readCsv("terro.csv")
    readCsv("trump.csv")
    with open('outtotal2.csv', 'a') as csvFile:
        fieldnames = ['user_id', 'created_at', 'hashtags', 'retweet_count', 'favorite_count', 'user_name','user_friends_count', 'user_followers_count', 'user_location', 'user_tweet_count', 'place', 'geo','tweets', 'happiness_index']
        writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
        writer.writeheader()
        for a,b,c,d,e,f,g,h,i,j,k,l,m,n in zip(uid, created_at, hashtags, retweet_count, favorite_count, user_name, user_friends_count, user_followers_count, user_location, user_tweet_count, place, geo, tweets, happ):
            thisT = {}
            thisT["user_id"] = a
            thisT["created_at"] = b
            thisT["hashtags"] = c
            thisT["retweet_count"] = d
            thisT["favorite_count"] = e
            thisT["user_name"] = f
            thisT["user_friends_count"] = g
            thisT["user_followers_count"] = h
            thisT["user_location"] = i
            thisT["user_tweet_count"] = j
            thisT["place"] = k
            thisT["geo"] = l
            thisT["tweets"] = m
            thisT["happiness_index"] = int(n)+2
            writer.writerow(thisT)





