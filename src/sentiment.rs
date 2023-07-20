use nitscrape::twt::{Tweet, TweetId};

use rust_bert::RustBertError;
use rust_bert::fnet::{FNetConfigResources, FNetModelResources, FNetVocabResources};
use rust_bert::pipelines::common::{ModelResource, ModelType};
use rust_bert::pipelines::sentiment::{SentimentConfig, SentimentModel, Sentiment, SentimentPolarity};
use rust_bert::resources::RemoteResource;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

use crate::stats::TimeSeriesItem;

pub struct Params<const N: usize> {
    values: [f32; N],
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProcessedTweet {
    pub tweet: Tweet,
    pub sentiment: Sentiment,
}

impl TimeSeriesItem for ProcessedTweet {
    fn timestamp(&self) -> DateTime<Utc> {
        self.tweet.timestamp
    }

    fn value(&self) -> f64 {
        if self.sentiment.polarity == SentimentPolarity::Negative {
            -self.sentiment.score
        } else {
            self.sentiment.score
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProcessedTweetRecord {
    pub id: TweetId,
    pub timestamp: DateTime<Utc>,
    pub username: String,
    pub text: String,
    pub lang: Option<nitscrape::twt::Language>,
    pub comments: u32,
    pub retweets: u32,
    pub quotes: u32,
    pub likes: u32,
    pub polarity: SentimentPolarity,
    pub score: f64,
}

impl From<ProcessedTweet> for ProcessedTweetRecord {
    fn from(value: ProcessedTweet) -> Self {
        Self::new(value.tweet, value.sentiment)
    }
}

impl ProcessedTweetRecord {
    pub fn new(tweet: Tweet, sent: Sentiment) -> Self {
        Self {
            id: tweet.id,
            timestamp: tweet.timestamp,
            username: tweet.username,
            text: tweet.text,
            lang: tweet.lang,
            comments: tweet.comments,
            retweets: tweet.retweets,
            quotes: tweet.quotes,
            likes: tweet.likes,
            polarity: sent.polarity,
            score: sent.score,
        }
    }

    pub fn processed_tweet(self) -> ProcessedTweet {
        ProcessedTweet { 
            tweet: Tweet {
                id: self.id,
                timestamp: self.timestamp,
                username: self.username,
                text: self.text,
                lang: self.lang,
                comments: self.comments,
                retweets: self.retweets,
                quotes: self.quotes,
                likes: self.likes,
            },
            sentiment: Sentiment { polarity: self.polarity, score: self.score }
        }
    }
}

pub struct TWBSentimentAnalyser {
    model: SentimentModel,
}

impl TWBSentimentAnalyser {
    const TWB_CONFIG: (&'static str, &'static str) = ("roberta-base/config", "model/config.json");
    const TWB_VOCAB: (&'static str, &'static str) = ("roberta-base/vocab", "model/vocab.json");
    const TWB_MODEL: (&'static str, &'static str) = ("roberta-base/model", "model/rust_model.ot");

    pub fn twitter_roberta_base() -> Result<Self, RustBertError> {
        //    Set-up classifier
        let config_resource = Box::new(RemoteResource::from_pretrained(
            Self::TWB_CONFIG,
        ));
        let vocab_resource = Box::new(RemoteResource::from_pretrained(
            Self::TWB_VOCAB,
        ));
        let model_resource = ModelResource::Torch(Box::new(RemoteResource::from_pretrained(
            Self::TWB_MODEL,
        )));

        let sentiment_config = SentimentConfig {
            model_type: ModelType::Roberta,
            model_resource,
            config_resource,
            vocab_resource,
            ..Default::default()
        };

        let model = SentimentModel::new(sentiment_config)?;

        Ok(
            TWBSentimentAnalyser { model }
        )
    }

    pub fn general_sentiment_model() -> Result<Self, RustBertError> {
        Ok(TWBSentimentAnalyser { model: SentimentModel::new(Default::default())? })
    }

    pub fn process_tweet(&self, tweet: Tweet) -> ProcessedTweet {
        let sentiment = self.analyse_tweet_text(&tweet.text);
        ProcessedTweet { tweet, sentiment }
    }

    pub fn analyse_tweet(&self, tweet: &Tweet) -> Sentiment {
        self.analyse_tweet_text(&tweet.text)
    }

    pub fn analyse_tweet_text(&self, txt: &str) -> Sentiment {
        self.model.predict([txt]).remove(0)
    }

    pub fn process_tweets<'a> (&'a self, tweets: Vec<Tweet>) -> Vec<ProcessedTweet> {
        let sentiments: Vec<Sentiment> = {
            let txts: Vec<&str> = tweets.iter().map(|t| t.text.as_ref()).collect();
            self.model.predict(txts)
        };
        
        tweets.into_iter().zip(sentiments.into_iter()).map(|(tweet, sentiment)| ProcessedTweet { tweet, sentiment }).collect()
    }
}