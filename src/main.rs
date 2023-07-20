use std::{time::{SystemTime}, env::args, collections::HashMap, fs::File, io::Write, process::Command};
use chrono::{NaiveDateTime, NaiveDate, DateTime, Utc};
use nitscrape::{table::{self, CsvLayout}, twt::TweetId, twt::Tweet};
use sentiment::ProcessedTweetRecord;
use serde::{Deserializer, Deserialize};
use stats::DataSeries;

use crate::stats::TimeSeriesItem;

pub mod sentiment;
pub mod stats;
pub mod tor_farm;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct IndexEntry {
    #[serde(deserialize_with = "naive_date_time_from_str")]
    #[serde(rename = "Created-At")]
    timestamp: NaiveDateTime,
    #[serde(rename = "From-User-Id")]
    uid: String,
    #[serde(rename = "To-User-Id")]
    toid: String,
    #[serde(rename = "Language")]
    lang: String,
    #[serde(rename = "Retweet-Count")]
    retw: f64,
    #[serde(rename = "PartyName")]
    party: String,
    #[serde(rename = "Id")]
    id: TweetId,
    #[serde(rename = "Score")]
    score: f64,
    #[serde(rename = "Scoring String")]
    score_str: String,
    #[serde(rename = "Negativity")]
    neg: f64,
    #[serde(rename = "Positivity")]
    pos: f64,
    #[serde(rename = "Uncovered Tokens")]
    tok: u32,
    #[serde(rename = "Total Tokens")]
    total_tok: u32,
}

fn naive_date_time_from_str<'de, D>(deserializer: D) -> Result<NaiveDateTime, D::Error>
where
    D: Deserializer<'de>,
{
    let s: String = Deserialize::deserialize(deserializer)?;
    NaiveDateTime::parse_from_str(&s, "%D %l:%M %p").map_err(serde::de::Error::custom)
}


mod brx {
    use nitscrape::twt::TweetId;

    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    pub enum Sentiment {
        #[serde(rename = "positive")]
        Positive,
        #[serde(rename = "neutral")]
        Neutral,
        #[serde(rename = "negative")]
        Negative,
    }
    
    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    pub enum Stance {
        #[serde(rename = "remain")]
        Remain,
        #[serde(rename = "leave")]
        Leave,
        #[serde(rename = "other")]
        Other,
    }
    

    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    pub struct Item {
        #[serde(rename = "ID")]
        id: TweetId,
        #[serde(rename = "user_id")]
        uid: String,
        #[serde(rename = "t_sentiment")]
        sentiment: Sentiment,
        #[serde(rename = "t_stance")]
        stance: Stance,
    }
}



#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let settings_path = args().nth(1).unwrap_or("settings.json".to_owned());

    if let Ok(settings_data) = std::fs::read_to_string(&settings_path) {
        //kill existing tor processes
        // let _ = Command::new("taskkill")
        //     .args(["/IM", "tor.exe", "/F"])
        //     .spawn().unwrap().wait();
        let settings: tor_farm::Settings = serde_json::from_str(&settings_data)?;
        println!("Hydrating from {} to {}", &settings.input_path, &settings.output_path);

        //hydrate(input, output, cursor, 1, 0, table::CsvLayout::without_timestamp(b'~', 0, Some(3))).await;
        //tor_farm::begin_tor_farm(input, table::CsvLayout::without_timestamp(b'~', 0, Some(3)), output, "C:\\Program Files\\Tor Browser\\Browser\\TorBrowser\\Tor\\tor.exe".to_owned(), "tor/torrc_base".to_owned(), 200, 0).await
        tor_farm::begin_tor_farm(settings, true).await
    } else {
        // No settings file... create it
        match std::fs::write(&settings_path, serde_json::to_string(&tor_farm::Settings::default()).unwrap()) {
            Ok(_) => println!("No settings file - a template file has been written to {}", &settings_path),
            Err(e) => println!("No settings file - failed to write a template file: {}", e),
        }
        
        Ok(())
    }

}


pub async fn hydrate(input: String, output: String, cursor: usize, sample_density: usize, attempts: usize, layout: CsvLayout) {
    let mut csv = table::TweetCsvReader::read_csv(input, layout).unwrap();
    let mut hydrator = csv.hydrator(output, cursor).unwrap();

    // Use a batch of tor proxies - we create torrc files for each proxy for each port. This allows us to use multiple circuits simultaneously.
    let tor_net_mgr = nitscrape::net::TorClientManager::generate_configs("C:\\Program Files\\Tor Browser\\Browser\\TorBrowser\\Tor\\tor.exe".to_owned(), "tor/torrc_base".to_owned(), 9000..9050).await.unwrap();
    //let tor_net_mgr = nitscrape::net::TorClientManager::from_generated_configs("C:\\Program Files\\Tor Browser\\Browser\\TorBrowser\\Tor\\tor.exe".to_owned(), 9000..9100).unwrap();

    for i in 0.. {
        if let Err(e) = hydrator.hydrate_batch(&mut tor_net_mgr.iter(), nitscrape::twt::BATCH_SIZE, sample_density, attempts).await {
            println!("Error at batch {}: {}", i, e);
            break;
        } else {
            println!("Hydrated batch {}, cursor: {}", i, hydrator.cursor());
        }
    }
}



/*
fn main() {
    show_existing_sentiment();
}
*/
pub fn analyse_sentiment() {
    let analyser = sentiment::TWBSentimentAnalyser::general_sentiment_model().unwrap();
    let sent = analyser.analyse_tweet_text("This is the worst day ever");
    print!("SENT: {:?}", sent);
}

pub fn compile_us_election_tweets() {
    let mut index_csv = csv::ReaderBuilder::new()
        .delimiter(b';')
        .from_path("tweets.csv").unwrap();

    let mut tweets_csv = csv::ReaderBuilder::new()
        .from_path("hydrated.csv").unwrap();

    let mut rep_writer = csv::WriterBuilder::new().from_path("rep.csv").unwrap();
    let mut dem_writer = csv::WriterBuilder::new().from_path("dem.csv").unwrap();

    let mut tweet_reader = tweets_csv.deserialize::<Tweet>();

    let mut tweet: Tweet = tweet_reader.next().unwrap().unwrap();

    let analyser = sentiment::TWBSentimentAnalyser::general_sentiment_model().unwrap();

    for (i, record) in index_csv.records().enumerate() {
        if i % 10000 == 0{
            println!("{} tweets processed", i);
        }
        if let Ok(record) = record {
            if let Some(id) = record.get(6).and_then(|x| x.parse::<TweetId>().ok()) {
                if &id == &tweet.id {
                    if let Some(topic) = record.get(5) {
                        if topic == "Republicans" {
                            rep_writer.serialize(ProcessedTweetRecord::from(analyser.process_tweet(tweet.clone()))).unwrap();
                        } else if topic == "Democrats" {
                            dem_writer.serialize(ProcessedTweetRecord::from(analyser.process_tweet(tweet.clone()))).unwrap();
                        }
                    }
                    if let Some(next) = tweet_reader.next().map(Result::unwrap) {
                        tweet = next;
                    }
                }
            }
        }
    }
}

pub fn plot_graphs() {
    let graph = stats::TweetSeries::from_processed_tweets_csv("rep.csv", NaiveDate::from_ymd_opt(2020, 7, 1).unwrap().and_hms_opt(12, 0, 0).unwrap().and_utc(), chrono::Duration::days(4)).unwrap();
    
    let mut total = 0;

    for (i, d) in graph.data.iter().enumerate() {
        total += d.len();
        println!("{}: len {}, mean {}", i, d.len(), d.iter().map(TimeSeriesItem::value).sum::<f64>() / d.len() as f64)
    }

    println!("TOTAL: {}", total);
    
    graph.linear_graph(Some(-1.0), Some(1.0), "US Election Tweet Sentiment", "Net Positivity", "sent_rep.png").unwrap();

    let common_words: HashMap<String, ()> = csv::ReaderBuilder::new().from_path("common_words.csv").unwrap().records().filter_map(Result::ok).map(|x| (x.get(0).unwrap().to_owned(), ())).collect();

    let mut table: Vec<Vec<String>> = Vec::new();

    for tweets in graph.data.iter() {
        let mut tmap = HashMap::<String, u32>::new();
        
        let mut row: Vec<String> = Vec::new();

        for ptweet in tweets.iter() {
            for word in ptweet.tweet.text.split_ascii_whitespace() {
                let w = word.to_owned().to_lowercase().replace(&['(', ')', ',', '\"', '.', ';', ':', '\'', '-', '&', '!', '?', '—', ' '], "");
                if !common_words.contains_key(&w) {
                    if let Some(counter) = tmap.get_mut(&w) {
                        *counter += 1;
                    } else {
                        if !w.is_empty() {
                            tmap.insert(w, 1);
                        }
                    }
                }
            }
        }
        // Sort to rank words.
        let mut vec = tmap.into_iter().collect::<Vec<(String, u32)>>();
        vec.sort_by_key(|x| -(x.1 as i32));

        'l: for (i, (w, n)) in vec.into_iter().enumerate() {
            if i < 30 {
                row.push(format!("{} ({}, {:.2}%)", w, n, (n as f32 / tweets.len() as f32) * 100.0));
            } else {
                break 'l;
            }
        }

        table.push(row);
    }

    let mut i: usize = 0;
    loop {
        let max_len = table.iter().filter_map(|x: &Vec<String>| x.get(i)).map(|x| x.chars().count()).max();
        if let Some(max_len) = max_len {
            for row in table.iter_mut() {
                if let Some(item) = row.get_mut(i) {
                    let diff = max_len - item.chars().count();
                    *item += ",";
                    *item += &vec![' '; diff].into_iter().collect::<String>();
                }
            }

            i += 1;
        } else {
            break;
        } 
    }

    let mut f = File::create("words.txt").unwrap();

    for row in table {
        for item in row {
            write!(f, "{} ", item);
        }
        write!(f, "\n");
    }

}

pub fn show_existing_sentiment() {
    let mut graph: DataSeries = stats::DataSeries::new(NaiveDate::from_ymd_opt(2020, 7, 1).unwrap().and_hms_opt(12, 0, 0).unwrap().and_utc(), chrono::Duration::days(1));

    let mut index_csv = csv::ReaderBuilder::new()
        .delimiter(b';')
        .from_path("tweets.csv").unwrap();

    for (i, item) in index_csv.deserialize::<IndexEntry>().enumerate() {
        if i % 10000 == 0 {
            println!("{} tweets processed", i);
        }
        let item = item.unwrap();
        if item.party == "BothParty" {
            let ns = item.pos - item.neg;
            graph.update(item.timestamp.and_utc(), ns);
        }
    }


    graph.linear_graph(Some(-1.0), Some(1.0), "US Election Tweet Sentiment", "Net Positivity", "idxsent_both.png").unwrap();

}