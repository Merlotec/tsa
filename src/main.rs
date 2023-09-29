use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};
use clap::Parser;
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use nitscrape::{
    table::{self, CsvLayout},
    twt::Tweet,
    twt::TweetId,
};
use sentiment::ProcessedTweetRecord;
use serde::{Deserialize, Deserializer};
use stats::DataSeries;
use std::{
    borrow::Cow,
    collections::HashMap,
    fs::{File, OpenOptions},
    io::Write,
    path::{PathBuf, Path},
    str::FromStr,
};

use crate::{stats::TimeSeriesItem, tor_farm::ResumeMethod};

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

#[derive(Debug, Default, Parser)]
struct DownloadArgs {
    #[clap(long, short = 't')]
    resume_from_tweet: bool,

    #[clap(long, short = 'f')]
    dont_resume: bool,

    #[clap(long, short = 'p')]
    show_progress: bool,

    #[clap(long, short = 's')]
    settings_path: Option<String>,
}

#[derive(Debug, Default, Parser)]
struct SentimentArgs {
    #[clap(long, short = 'i')]
    input_path: String,

    #[clap(long, short = 'o')]
    output_path: Option<String>,

    #[clap(long, short = 'f')]
    dont_resume: bool,

    #[clap(long, short = 'p')]
    show_progress: bool,
}

#[derive(Debug, Default, Parser)]
struct PlotSentimentArgs {
    #[clap(long, short = 'o', default_value = "graph.png")]
    output_path: String,

    #[clap(long, short = 'i', num_args = 1.., value_delimiter = ' ')]
    input_paths: Vec<String>,

    #[clap(long, short = 'f', default_value = "%d-%m-%Y")]
    date_format: String,

    #[clap(long, short = 't')]
    title: Option<String>,

    #[clap(long, short = 'y')]
    y_desc: Option<String>,

    #[clap(long, short = 's')]
    start: Option<String>,

    #[clap(long, short = 'd', default_value = "4")]
    interval: i64,

    #[clap(long, short = 'p')]
    comparison_plot: Option<String>,

    #[clap(long, short = 'e')]
    export_path: Option<String>,
}

#[derive(Debug, Default, Parser)]
struct WordPlotArgs {
    #[clap(long, short = 'o', default_value = "words_graph.png")]
    output_path: String,

    #[clap(long, short = 'i', num_args = 1.., value_delimiter = ' ')]
    input_paths: Vec<String>,

    #[clap(long, short = 'w', num_args = 1.., value_delimiter = ' ')]
    words: Vec<String>,

    #[clap(long, short = 'n')]
    normalise: bool,

    #[clap(long, short = 'f', default_value = "%d-%m-%Y")]
    date_format: String,

    #[clap(long, short = 't')]
    title: Option<String>,

    #[clap(long, short = 'y')]
    y_desc: Option<String>,

    #[clap(long, short = 's')]
    start: Option<String>,

    #[clap(long, short = 'd', default_value = "4")]
    interval: i64,

    #[clap(long, short = 'p')]
    comparison_plot: Option<String>,

    #[clap(long, short = 'e')]
    export_path: Option<String>,
}

#[derive(Debug, Default, Parser)]
struct SentimentWordPlotArgs {
    #[clap(long, short = 'o', default_value = "words_graph.png")]
    output_path: String,

    #[clap(long, short = 'i', num_args = 1.., value_delimiter = ' ')]
    input_paths: Vec<String>,

    #[clap(long, short = 'w', num_args = 1.., value_delimiter = ' ')]
    words: Vec<String>,

    #[clap(long, short = 'f', default_value = "%d-%m-%Y")]
    date_format: String,

    #[clap(long, short = 't')]
    title: Option<String>,

    #[clap(long, short = 'y')]
    y_desc: Option<String>,

    #[clap(long, short = 's')]
    start: Option<String>,

    #[clap(long, short = 'd', default_value = "4")]
    interval: i64,

    #[clap(long, short = 'p')]
    comparison_plot: Option<String>,

    #[clap(long, short = 'e')]
    export_path: Option<String>,
}

#[derive(Debug, Default, Parser)]
struct WordsArgs {
    #[clap(long, short = 'i', num_args = 1.., value_delimiter = ' ')]
    input_paths: Vec<String>,

    #[clap(long, short = 'o', default_value = "words.csv")]
    output_path: String,

    #[clap(long, short = 'x', default_value = "ignore.csv")]
    ignore_path: String,

    #[clap(long, short = 'n', default_value = "20")]
    num_words: i32,

    #[clap(long, short = 's')]
    start: Option<String>,

    #[clap(long, short = 'v', default_value = "4")]
    interval: i64,

    #[clap(long, short = 'p')]
    pretty: bool,
}

#[derive(Debug, Default, Parser)]
pub struct CountArgs {
    #[arg(global = true)]
    path: String,

    #[clap(long, short = 'h')]
    has_headers: bool,
}

#[derive(Debug, Parser)]
#[clap(name = "tsa", version, author = "Brodie Knight")]
enum TsaCommand {
    Count(CountArgs),
    Download(DownloadArgs),
    Sentiment(SentimentArgs),
    PlotSentiment(PlotSentimentArgs),
    PlotWords(WordPlotArgs),
    PlotWordSentiment(SentimentWordPlotArgs),
    Words(WordsArgs),
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cmd: TsaCommand = TsaCommand::parse();

    match cmd {
        TsaCommand::Count(args) => {
            let mut csv = csv::ReaderBuilder::new().has_headers(args.has_headers).from_path(&args.path)?;
            println!("{} has {} records (has_headers={})", args.path, csv.records().count(), args.has_headers);
            Ok(())
        }
        TsaCommand::Download(args) => {
            let settings_path = args.settings_path.unwrap_or("settings.json".to_owned());

            if let Ok(settings_data) = std::fs::read_to_string(&settings_path) {
                //kill existing tor processes
                // let _ = Command::new("taskkill")
                //     .args(["/IM", "tor.exe", "/F"])
                //     .spawn().unwrap().wait();
                let settings: tor_farm::Settings = serde_json::from_str(&settings_data)?;
                println!(
                    "Hydrating from {} to {}",
                    &settings.input_path, &settings.output_path
                );

                //hydrate(input, output, cursor, 1, 0, table::CsvLayout::without_timestamp(b'~', 0, Some(3))).await;
                //tor_farm::begin_tor_farm(input, table::CsvLayout::without_timestamp(b'~', 0, Some(3)), output, "C:\\Program Files\\Tor Browser\\Browser\\TorBrowser\\Tor\\tor.exe".to_owned(), "tor/torrc_base".to_owned(), 200, 0).await

                let resume = if args.dont_resume {
                    ResumeMethod::None
                } else {
                    if args.resume_from_tweet {
                        ResumeMethod::TweetId
                    } else {
                        ResumeMethod::ResumeFile
                    }
                };
                tor_farm::begin_tor_farm(settings, resume, args.show_progress).await
            } else {
                // No settings file... create it
                match std::fs::write(
                    &settings_path,
                    serde_json::to_string(&tor_farm::Settings::default()).unwrap(),
                ) {
                    Ok(_) => println!(
                        "No settings file - a template file has been written to {}",
                        &settings_path
                    ),
                    Err(e) => println!("No settings file - failed to write a template file: {}", e),
                }

                Ok(())
            }
        }
        TsaCommand::Sentiment(args) => {
            let mut input = csv::ReaderBuilder::new().from_path(&args.input_path)?;

            let out_path = args
                .output_path
                .or_else(|| default_output_path(&args.input_path, "_out"))
                .ok_or("Could not determine default output path!")?;

            let start_id: Option<TweetId> = {
                if args.dont_resume {
                    None
                } else {
                    // Try read output
                    csv::ReaderBuilder::new()
                        .from_path(&out_path)
                        .ok()
                        .and_then(|mut r| {
                            r.deserialize::<Tweet>()
                                .last()
                                .map(|x| x.expect("Last element corrupted!").id)
                        })
                }
            };

            let output_file = OpenOptions::new()
                .write(true)
                .append(!args.dont_resume)
                .create(true)
                .open(&out_path)?;

            let mut output = csv::Writer::from_writer(output_file);

            let model = tokio::task::spawn_blocking(|| {
                sentiment::TWBSentimentAnalyser::general_sentiment_model().unwrap()
            })
            .await?;

            let mut i = 0;
            // Get cursor up to position
            let mut reader = input.deserialize::<Tweet>();
            if let Some(start_id) = start_id {
                for j in 0.. {
                    if let Some(Ok(t)) = reader.next() {
                        if t.id == start_id {
                            i = j;
                            println!("Found tweet at index {}.", j);
                            break;
                        }
                    }
                }
            }

            let pb = {
                if args.show_progress {
                    // We need to read the length of the input:
                    println!("Reading input csv file for progress bar (this may take a while)...");
                    let bar = ProgressBar::new(
                        csv::ReaderBuilder::new()
                            .from_path(&args.input_path)?
                            .records()
                            .count() as u64,
                    );
                    bar.set_style(ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})").unwrap().with_key("eta", |state: &ProgressState, w: &mut dyn std::fmt::Write| write!(w, "{:.1}min", state.eta().as_secs_f64() / 60.0).unwrap()));
                    Some(bar)
                } else {
                    None
                }
            };

            for tweet in reader {
                if let Ok(tweet) = tweet {
                    if tweet.lang == Some(nitscrape::twt::Language::English) {
                        let record = ProcessedTweetRecord::from(model.process_tweet(tweet));
                        let _ = output.serialize(record);
                    }
                }
                i += 1;
                if let Some(pb) = &pb {
                    pb.set_position(i as u64);
                }
            }
            Ok(())
        }
        TsaCommand::PlotSentiment(args) => {
            let interval_days = args.interval;
            let graph = if let Some(start) = args.start {
                match chrono::NaiveDate::parse_from_str(&start, "%d-%m-%Y") {
                    Ok(x) => {
                        let dt = chrono::DateTime::<chrono::Utc>::from_utc(
                            x.and_hms_opt(12, 0, 0).unwrap(),
                            chrono::Utc,
                        );
                        stats::TweetSeries::from_processed_tweets_csv(
                            &args.input_paths,
                            Some(dt),
                            chrono::Duration::days(interval_days),
                        )
                        .unwrap()
                    }
                    Err(e) => {
                        panic!("Failed to parse start date: {}", e);
                    }
                }
            } else {
                stats::TweetSeries::from_processed_tweets_csv(
                    &args.input_paths,
                    None,
                    chrono::Duration::days(interval_days),
                )
                .unwrap()
            };

            // let mut total = 0;

            // for (i, d) in graph.data.iter().enumerate() {
            //     total += d.len();
            //     println!(
            //         "{}: len {}, mean {}",
            //         i,
            //         d.len(),
            //         d.iter().map(TimeSeriesItem::value).sum::<f64>() / d.len() as f64
            //     )
            // }

            let series = graph.ave_series_dt();

            if let Some(export_path) = args.export_path {
                if let Err(e) = stats::export(series.iter().copied(), export_path, &args.date_format) {
                    println!("Failed to export data: {}", e);
                }
            }

            // println!("TOTAL: {}", total);

            let comparison_series: Option<Vec<(chrono::DateTime<chrono::Utc>, f64)>> = args
                .comparison_plot
                .map(|x| comparison_data(x).expect("Failed to fetch comparison data"));

            stats::linear_graph(
                series,
                comparison_series,
                graph.start_date(),
                graph.end_date(),
                Some(-1.0),
                Some(1.0),
                &args.date_format,
                args.title.unwrap_or_default(),
                args.y_desc.unwrap_or_default(),
                &args.output_path,
            )
            .unwrap();

            Ok(())
        }
        TsaCommand::PlotWords(args) => {
            let interval_days = args.interval;
            let graph = if let Some(start) = args.start {
                match chrono::NaiveDate::parse_from_str(&start, "%d-%m-%Y") {
                    Ok(x) => {
                        let dt = chrono::DateTime::<chrono::Utc>::from_utc(
                            x.and_hms_opt(12, 0, 0).unwrap(),
                            chrono::Utc,
                        );
                        stats::TweetSeries::from_processed_tweets_csv_with_words(
                            &args.input_paths,
                            Some(dt),
                            chrono::Duration::days(interval_days),
                            &args.words,
                        )
                        .unwrap()
                    }
                    Err(e) => {
                        panic!("Failed to parse start date: {}", e);
                    }
                }
            } else {
                stats::TweetSeries::from_processed_tweets_csv_with_words(
                    &args.input_paths,
                    None,
                    chrono::Duration::days(interval_days),
                    &args.words,
                )
                .unwrap()
            };

            let comparison_series: Option<Vec<(chrono::DateTime<chrono::Utc>, f64)>> = args
                .comparison_plot
             .map(|x| comparison_data(x).expect("Failed to fetch comparison data"));


            let series = if args.normalise {
                graph.ave_series_dt()
            } else {
                graph.sum_series_dt()
            };

            if let Some(export_path) = args.export_path {
                if let Err(e) = stats::export(series.iter().copied(), export_path, &args.date_format) {
                    println!("Failed to export data: {}", e);
                }
            }

            stats::linear_graph(
                series,
                comparison_series,
                graph.start_date(),
                graph.end_date(),
                None,
                None,
                &args.date_format,
                args.title.unwrap_or_default(),
                args.y_desc.unwrap_or_default(),
                &args.output_path,
            )
            .unwrap();

            Ok(())
        }
        TsaCommand::PlotWordSentiment(args) => {
            let interval_days = args.interval;
            let graph = if let Some(start) = args.start {
                match chrono::NaiveDate::parse_from_str(&start, "%d-%m-%Y") {
                    Ok(x) => {
                        let dt = chrono::DateTime::<chrono::Utc>::from_utc(
                            x.and_hms_opt(12, 0, 0).unwrap(),
                            chrono::Utc,
                        );
                        stats::TweetSeries::from_processed_tweets_csv_sentiment_with_words(
                            &args.input_paths,
                            Some(dt),
                            chrono::Duration::days(interval_days),
                            &args.words,
                        )
                        .unwrap()
                    }
                    Err(e) => {
                        panic!("Failed to parse start date: {}", e);
                    }
                }
            } else {
                stats::TweetSeries::from_processed_tweets_csv_sentiment_with_words(
                    &args.input_paths,
                    None,
                    chrono::Duration::days(interval_days),
                    &args.words,
                )
                .unwrap()
            };

            let comparison_series: Option<Vec<(chrono::DateTime<chrono::Utc>, f64)>> = args
                .comparison_plot
             .map(|x| comparison_data(x).expect("Failed to fetch comparison data"));

             let series = graph.ave_series_dt();

             if let Some(export_path) = args.export_path {
                 if let Err(e) = stats::export(series.iter().copied(), export_path, &args.date_format) {
                     println!("Failed to export data: {}", e);
                 }
             }

            stats::linear_graph(
                series,
                comparison_series,
                graph.start_date(),
                graph.end_date(),
                Some(-1.0),
                Some(1.0),
                &args.date_format,
                args.title.unwrap_or_default(),
                args.y_desc.unwrap_or_default(),
                &args.output_path,
            )
            .unwrap();

            Ok(())
        }
        TsaCommand::Words(args) => {
            let interval_days = args.interval;
            let graph = if let Some(start) = args.start {
                match chrono::NaiveDate::parse_from_str(&start, "%d-%m-%Y") {
                    Ok(x) => {
                        let dt = chrono::DateTime::<chrono::Utc>::from_utc(
                            x.and_hms_opt(12, 0, 0).unwrap(),
                            chrono::Utc,
                        );
                        stats::TweetSeries::from_processed_tweets_csv(
                            &args.input_paths,
                            Some(dt),
                            chrono::Duration::days(interval_days),
                        )
                        .unwrap()
                    }
                    Err(e) => {
                        panic!("Failed to parse start date: {}", e);
                    }
                }
            } else {
                stats::TweetSeries::from_processed_tweets_csv(
                    &args.input_paths,
                    None,
                    chrono::Duration::days(interval_days),
                )
                .unwrap()
            };

            let common_words: HashMap<String, ()> = csv::ReaderBuilder::new()
                .from_path(&args.ignore_path)
                .unwrap()
                .records()
                .filter_map(Result::ok)
                .map(|x| (x.get(0).unwrap().to_owned(), ()))
                .collect();

            let mut table: Vec<Vec<String>> = Vec::new();

            let mut max_count: usize = 0;

            for tweets in graph.data.iter() {
                if tweets.len() > max_count {
                    max_count = tweets.len();
                }

                let mut tmap = HashMap::<String, u32>::new();

                let mut row: Vec<String> = Vec::new();

                for ptweet in tweets.iter() {
                    for word in ptweet.tweet.text.split_ascii_whitespace() {
                        let w = word
                            .to_owned()
                            .to_lowercase()
                            .replace(&nitscrape::twt::JUNK_CHARACTERS, "");
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
                        if args.pretty {
                            row.push(format!(
                                "{} ({} | {:.4})",
                                w,
                                n,
                                (n as f32 / tweets.len() as f32)
                            ));
                        } else {
                            row.push(format!(
                                "{};{};{:.4}",
                                w,
                                n,
                                (n as f32 / tweets.len() as f32)
                            ));
                        }
                        
                    } else {
                        break 'l;
                    }
                }

                table.push(row);
            }

            if args.pretty {
                let mut i: usize = 0;
                loop {
                    let max_len = table
                        .iter()
                        .filter_map(|x: &Vec<String>| x.get(i))
                        .map(|x| x.chars().count())
                        .max();
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
            }

            let count_width = max_count.to_string().len();

            let mut f = File::create(&args.output_path).unwrap();

            for (i, row) in table.iter().enumerate() {
                let _ = write!(f, "{},", graph.date_for(i).format("%d-%m-%Y"));

                if args.pretty {
                    let _ = write!(f, "{:>width$}, ", graph.data[i].len(), width=count_width);
                } else {
                    let _ = write!(f, "{},", graph.data[i].len());
                }

                for item in row {
                    if args.pretty {
                        let _ = write!(f, "{} ", item);
                    } else {
                        let _ = write!(f, "{},", item);
                    }
                    
                }
                let _ = write!(f, "\n");
            }

            Ok(())
        }
    }
}

pub async fn hydrate(
    input: String,
    output: String,
    cursor: usize,
    sample_density: usize,
    attempts: usize,
    layout: CsvLayout,
) {
    let mut csv = table::TweetCsvReader::read_csv(input, layout).unwrap();
    let mut hydrator = csv.hydrator(output, cursor).unwrap();

    // Use a batch of tor proxies - we create torrc files for each proxy for each port. This allows us to use multiple circuits simultaneously.
    let tor_net_mgr = nitscrape::net::TorClientManager::generate_configs(
        "C:\\Program Files\\Tor Browser\\Browser\\TorBrowser\\Tor\\tor.exe".to_owned(),
        "tor/torrc_base".to_owned(),
        9000..9050,
        30.0,
    )
    .await
    .unwrap();
    //let tor_net_mgr = nitscrape::net::TorClientManager::from_generated_configs("C:\\Program Files\\Tor Browser\\Browser\\TorBrowser\\Tor\\tor.exe".to_owned(), 9000..9100).unwrap();

    for i in 0.. {
        if let Err(e) = hydrator
            .hydrate_batch(
                &mut tor_net_mgr.iter(),
                nitscrape::twt::BATCH_SIZE,
                sample_density,
                attempts,
            )
            .await
        {
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
        .from_path("tweets.csv")
        .unwrap();

    let mut tweets_csv = csv::ReaderBuilder::new().from_path("hydrated.csv").unwrap();

    let mut rep_writer = csv::WriterBuilder::new().from_path("rep.csv").unwrap();
    let mut dem_writer = csv::WriterBuilder::new().from_path("dem.csv").unwrap();

    let mut tweet_reader = tweets_csv.deserialize::<Tweet>();

    let mut tweet: Tweet = tweet_reader.next().unwrap().unwrap();

    let analyser = sentiment::TWBSentimentAnalyser::general_sentiment_model().unwrap();

    for (i, record) in index_csv.records().enumerate() {
        if i % 10000 == 0 {
            println!("{} tweets processed", i);
        }
        if let Ok(record) = record {
            if let Some(id) = record.get(6).and_then(|x| x.parse::<TweetId>().ok()) {
                if &id == &tweet.id {
                    if let Some(topic) = record.get(5) {
                        if topic == "Republicans" {
                            rep_writer
                                .serialize(ProcessedTweetRecord::from(
                                    analyser.process_tweet(tweet.clone()),
                                ))
                                .unwrap();
                        } else if topic == "Democrats" {
                            dem_writer
                                .serialize(ProcessedTweetRecord::from(
                                    analyser.process_tweet(tweet.clone()),
                                ))
                                .unwrap();
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
    let graph = stats::TweetSeries::from_processed_tweets_csv(
        &["rep.csv"],
        Some(
            NaiveDate::from_ymd_opt(2016, 7, 1)
                .unwrap()
                .and_hms_opt(12, 0, 0)
                .unwrap()
                .and_utc(),
        ),
        chrono::Duration::days(4),
    )
    .unwrap();

    let mut total = 0;

    for (i, d) in graph.data.iter().enumerate() {
        total += d.len();
        println!(
            "{}: len {}, mean {}",
            i,
            d.len(),
            d.iter().map(TimeSeriesItem::value).sum::<f64>() / d.len() as f64
        )
    }

    println!("TOTAL: {}", total);

    stats::linear_graph(
        graph.ave_series_dt(),
        None,
        graph.start_date(),
        graph.end_date(),
        Some(-1.0),
        Some(1.0),
        "%Y-%M",
        "Brexit Tweet Sentiment",
        "Net Positivity",
        "sent_brx.png",
    )
    .unwrap();

    let common_words: HashMap<String, ()> = csv::ReaderBuilder::new()
        .from_path("common_words.csv")
        .unwrap()
        .records()
        .filter_map(Result::ok)
        .map(|x| (x.get(0).unwrap().to_owned(), ()))
        .collect();

    let mut table: Vec<Vec<String>> = Vec::new();

    for tweets in graph.data.iter() {
        let mut tmap = HashMap::<String, u32>::new();

        let mut row: Vec<String> = Vec::new();

        for ptweet in tweets.iter() {
            for word in ptweet.tweet.text.split_ascii_whitespace() {
                let w = word
                    .to_owned()
                    .to_lowercase()
                    .replace(&nitscrape::twt::JUNK_CHARACTERS, "");
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
                row.push(format!(
                    "{} ({}; {:.2}%)",
                    w,
                    n,
                    (n as f32 / tweets.len() as f32) * 100.0
                ));
            } else {
                break 'l;
            }
        }

        table.push(row);
    }

    let mut i: usize = 0;
    loop {
        let max_len = table
            .iter()
            .filter_map(|x: &Vec<String>| x.get(i))
            .map(|x| x.chars().count())
            .max();
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

    let mut f = File::create("words.csv").unwrap();

    for row in table {
        for item in row {
            let _ = write!(f, "{} ", item);
        }
        let _ = write!(f, "\n");
    }
}

pub fn show_existing_sentiment() {
    let mut graph: DataSeries = stats::DataSeries::new(
        NaiveDate::from_ymd_opt(2020, 7, 1)
            .unwrap()
            .and_hms_opt(12, 0, 0)
            .unwrap()
            .and_utc(),
        chrono::Duration::days(1),
    );

    let mut index_csv = csv::ReaderBuilder::new()
        .delimiter(b';')
        .from_path("tweets.csv")
        .unwrap();

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

    graph
        .linear_graph(
            Some(-1.0),
            Some(1.0),
            "US Election Tweet Sentiment",
            "Net Positivity",
            "idxsent_both.png",
        )
        .unwrap();
}

fn default_output_path(base: &str, append: &str) -> Option<String> {
    let pb = PathBuf::from_str(base).ok()?;
    let dir = pb
        .parent()
        .map(|x| x.to_string_lossy())
        .unwrap_or(Cow::Borrowed(""));
    let stem = pb.file_stem()?.to_string_lossy();
    let mut f: String = stem.into_owned() + append;
    if let Some(ext) = pb.extension().map(|x| x.to_string_lossy()) {
        f += ".";
        f += ext.as_ref();
    }

    let mut buf = PathBuf::from_str(&dir).ok()?;
    buf.push(f);

    Some(buf.to_string_lossy().into())
}

pub fn comparison_data(path: impl AsRef<Path>) -> csv::Result<Vec<(DateTime<Utc>, f64)>> {
    let mut csv = csv::ReaderBuilder::new().from_path(path)?;

    let mut comparison_data: Vec<(chrono::DateTime<chrono::Utc>, f64)> = Vec::new();
    for record in csv.records() {
        // Record 1 is date:

        if let Ok(record) = record {
            let mut date: Option<chrono::DateTime<chrono::Utc>> = None;
            if let Some(date_rcd) = record.get(0) {
                match chrono::NaiveDate::parse_from_str(&date_rcd, "%d-%m-%Y") {
                    Ok(x) => {
                        let dt = chrono::DateTime::<chrono::Utc>::from_utc(
                            x.and_hms_opt(12, 0, 0).unwrap(),
                            chrono::Utc,
                        );
                        date = Some(dt);
                    }
                    Err(e) => {
                        panic!("Failed to parse start date: {}", e);
                    }
                }
            }

            let value: Option<f64> = record.get(1).and_then(|x| x.parse::<f64>().ok());

            if let (Some(date), Some(value)) = (date, value) {
                comparison_data.push((date, value));
            }
        }
    }

    Ok(comparison_data)
}
