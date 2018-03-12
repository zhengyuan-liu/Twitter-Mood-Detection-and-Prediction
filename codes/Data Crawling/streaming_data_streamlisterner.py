import tweepy
from tweepy.streaming import StreamListener
from tweepy import Stream
import json
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import time

consumer_key = "JH0I2tWncc75pdj6kXvWBSkr7"
consumer_secret = "mmo0gIY8pLZwR1iFoTCp89vpn2TVCWiHOmXOTSC5cnfMbFZkzN"
access_token = "865302386-PHx5eiQFjfqtLCT9Lq6VcnUydnGLZGPV8nbugwCH"
access_token_secret = "98oi4IaoV3D1eHyNIWR2MJ7hVSOYamuiOkmNyy357UD9R"

# consumer_key = "fgL6eMWPateArS7q7XLwHTgMk"
# consumer_secret = "lbtJjlPuu5OrlnZYCrt1fOLMjk9iR8csyCnYJgGNan2EZlSK0Y"
# access_token = "865302386-07FD3e3IGnD6fSkcwEoGTfPJj7ZSHJnIspuKDHGR"
# access_token_secret = "m2U0sg9d1SkCq1foAblE6bdHi5v8RK6GcpnDygOrJJ0iq"

def process_status(sta):
	print sta.user
	print sta.text

tweetsDict = {}

class StreamOutListener(StreamListener):
    # def __init__(self, totalCount):
    #     StreamListener.__init__(self)
    #     self.totalCount = totalCount
    def on_error(self, status_code):
        print(status_code)
        if status_code == 420:  
            #returning False in on_data disconnects the stream  
            return False 

    def on_status(self,status):
        #print(status)
        global total_count
        if(total_count>=0):
            tweetsDict[status.id_str] = {}
            thisTweet = tweetsDict[status.id_str]
            thisTweet["user_id"] = status.user.id_str
            thisTweet["created_at"] = str(status.created_at)
            thisTweet["hashtags"] = []
            for hashtag in status.entities["hashtags"]:
                thisTweet["hashtags"].append(hashtag["text"])
            thisTweet["retweet_count"] = status.retweet_count
            thisTweet["favorite_count"] = status.favorite_count
            thisTweet["user_name"] = status.user.name
            thisTweet["user_friends_count"] = status.user.friends_count
            thisTweet["user_followers_count"] = status.user.followers_count
            thisTweet["user_location"] = status.user.location
            thisTweet["user_tweet_count"] = status.user.statuses_count
            thisTweet["place"] = str(status.place)
            thisTweet["geo"] = str(status.geo)
            thisTweet["tweets"] = status.text.encode('utf-8')

            with open('cs145data_filtered_motherday.json', 'w') as outFile:
                json.dump(tweetsDict, outFile, sort_keys=True)
        total_count -=1
        #print("total count %d" %total_count)


if __name__ == "__main__":
    #pdb.set_trace()

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    listener = StreamOutListener()
    stream = Stream(auth=auth, listener = listener)
    global total_count
    total_count = 3000
    #total_count
    # Use LA's coordinate and keyword of event to filter out results
    stream.filter(track = ["valentine"],languages = ["en"], locations = [-118.42, 33.9035, -118.0824, 34.1357],async = True)




		
