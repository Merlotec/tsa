use chrono::{DateTime, NaiveDateTime, Utc};
use clap::Parser;
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use nitscrape::{
    twt::Tweet,
    twt::TweetId,
};
use sentiment::ProcessedTweetRecord;
use serde::{Deserialize, Deserializer};
use std::{
    borrow::Cow,
    collections::HashMap,
    fs::{File, OpenOptions},
    io::Write,
    path::{PathBuf, Path},
    str::FromStr,
};

use crate::tor_farm::ResumeMethod;

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

    #[clap(long, short = 'b', default_value = "1024")]
    breadth: u32,

    #[clap(long, short = 'h', default_value = "768")]
    height: u32,

    #[clap(long, short = 'l', default_value = "12")]
    label_size: u32,

    #[clap(long, default_value = "2")]
    stroke_width: u32,
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

    #[clap(long, short = 'b', default_value = "1024")]
    breadth: u32,

    #[clap(long, short = 'h', default_value = "768")]
    height: u32,

    #[clap(long, short = 'l', default_value = "12")]
    label_size: u32,

    #[clap(long, default_value = "2")]
    stroke_width: u32,
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

    #[clap(long, short = 'b', default_value = "1024")]
    breadth: u32,

    #[clap(long, short = 'h', default_value = "768")]
    height: u32,

    #[clap(long, short = 'l', default_value = "12")]
    label_size: u32,

    #[clap(long, default_value = "2")]
    stroke_width: u32,
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
                args.breadth,
                args.height,
                args.label_size,
                args.stroke_width,
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
                args.breadth,
                args.height,
                args.label_size,
                args.stroke_width,
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
                args.breadth,
                args.height,
                args.label_size,
                args.stroke_width,
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
