use std::error::Error;

use chrono::{DateTime, Utc};
use plotters::prelude::*;

use crate::sentiment::{ProcessedTweet, ProcessedTweetRecord};

#[derive(Debug, Copy, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DataPoint {
    pub time: DateTime<Utc>,
    pub val: f64,
}

impl TimeSeriesItem for DataPoint {
    fn timestamp(&self) -> DateTime<Utc> {
        self.time
    }
    fn value(&self) -> f64 {
        self.val
    }
}

pub trait TimeSeriesItem {
    fn timestamp(&self) -> DateTime<Utc>;
    fn value(&self) -> f64;
}

pub struct DataSeries {
    start: DateTime<Utc>,
    interval: chrono::Duration,
    pub data: Vec<(f64, usize)>,
}

impl DataSeries {
    pub fn new(start: DateTime<Utc>, interval: chrono::Duration) -> Self {
        Self {
            start,
            interval,
            data: Vec::new(),
        }
    }

    /// Spits the tweet back out of not inserted.
    pub fn update(&mut self, timestamp: DateTime<Utc>, v: f64) {
        let dur = timestamp.signed_duration_since(self.start);
        if dur.num_milliseconds() < 0 {
            return;
        }

        let idx = (dur.num_milliseconds() / self.interval.num_milliseconds()) as usize;

        if self.data.len() > idx {
            let (old, i) = self.data[idx];
            if i == 0 {
                self.data[idx] = (v, 1); // Gets rid of 0
            } else {
                let new_i = i + 1;
                let delta = v - old;
                let new: f64 = old + (delta / new_i as f64);
                self.data[idx] = (new, new_i);
            }   
        } else {
            for _ in self.data.len()..idx {
                self.data.push((0.0, 0)); // Fill in data points in between, note 0 should never actually be shown outside.
            }
            self.data.push((v, 1));
        }
    }

    pub fn start_date(&self) -> DateTime<Utc> {
        self.start
    }

    pub fn series_days(&self) -> Vec<(f64, f64)> {
        self.data.iter().enumerate().filter_map(|(i, (v, s))| if *s == 0 { None } else { Some(((self.interval * i as i32).num_days() as f64, *v)) }).collect()
    }

    // This only works if the interval is at least a day.
    pub fn series_dt(&self) -> Vec<(DateTime<Utc>, f64)> {
        self.data.iter().enumerate().filter_map(|(i, (v, s))| if *s == 0 { None } else { Some((self.start + (self.interval * i as i32), *v)) }).collect()
    }

    pub fn end_date(&self) -> DateTime<Utc> {
        self.start + (self.interval * self.data.len() as i32)
    }

    pub fn linear_graph<P: AsRef<std::path::Path>>(&self, y_min: Option<f64>, y_max: Option<f64>, caption: impl AsRef<str>, y_desc: impl Into<String>, path: P) -> Result<(), Box<dyn Error>> {
        let root = BitMapBackend::new(&path, (1024, 768)).into_drawing_area();

        root.fill(&WHITE)?;
    
        let series = self.series_dt();

        let mut min = series.iter().map(|(_, v)| *v).reduce(f64::min).unwrap();
        let mut max = series.iter().map(|(_, v)| *v).reduce(f64::max).unwrap();

        if let Some(v) = y_min {
            min = v;
        }

        if let Some(v) = y_max {
            max = v;
        }

        let pad = (max - min) * 0.1;

        let mut chart = ChartBuilder::on(&root)
            .margin(10)
            .caption(
                caption,
                ("sans-serif", 40),
            )
            .set_label_area_size(LabelAreaPosition::Left, 60)
            .set_label_area_size(LabelAreaPosition::Right, 60)
            .set_label_area_size(LabelAreaPosition::Bottom, 40)
            .build_cartesian_2d(
                self.start..self.end_date(),
                min - pad..max + pad,
            )?;

        chart
            .configure_mesh()
            .disable_x_mesh()
            .disable_y_mesh()
            .x_labels(30)
            .max_light_lines(4)
            .y_desc(y_desc)
            .draw()?;

        chart.draw_series(LineSeries::new(
            series.into_iter(),
            &BLUE,
        ))?;

        // chart.draw_series(
        //     DATA.iter()
        //         .map(|(y, m, t)| Circle::new((Utc.ymd(*y, *m, 1), *t), 3, BLUE.filled())),
        // )?;

        // To avoid the IO failure being ignored silently, we manually call the present function
        root.present()?;

        Ok(())
    }
}

pub struct TweetSeries<T: TimeSeriesItem> {
    start: DateTime<Utc>,
    interval: chrono::Duration,
    pub data: Vec<Vec<T>>,
}

impl TweetSeries<DataPoint> {
    pub fn from_processed_tweets_csv_with_words<P: AsRef<std::path::Path>>(paths: &[P], start: Option<DateTime<Utc>>, interval: chrono::Duration, words: &[String]) -> csv::Result<Self> {
        // Find the start

        let mut tbuf: Option<DateTime<Utc>> = start;
        if tbuf.is_none() {
            for path in paths {
                let mut tweets_csv = csv::ReaderBuilder::new()
                    .from_path(&path)?;
                for tweet in tweets_csv.deserialize::<ProcessedTweetRecord>() {
                    if let Ok(tweet) = tweet.map(ProcessedTweetRecord::processed_tweet) {
                        let ts = tweet.timestamp();
                        match tbuf {
                            Some(t) => if ts < t { tbuf = Some(ts) },
                            None => tbuf = Some(ts),
                        }
                    }
                }
            }
        }


        let mut series = Self::new(tbuf.expect("No start date found!"), interval);
        
        for path in paths {
            let mut tweets_csv = csv::ReaderBuilder::new()
                .from_path(&path)?;
            for tweet in tweets_csv.deserialize::<ProcessedTweetRecord>() {
                if let Ok(tweet) = tweet.map(ProcessedTweetRecord::processed_tweet) {
                    let tw_words = tweet.tweet.processed_words(&[]);
                    let n = tw_words.iter().filter(|x| words.contains(x)).count();
                    series.insert(DataPoint { time: tweet.timestamp(), val: n as f64 });
                }
            }
        }


        Ok(series)
    }

    pub fn from_processed_tweets_csv_sentiment_with_words<P: AsRef<std::path::Path>>(paths: &[P], start: Option<DateTime<Utc>>, interval: chrono::Duration, words: &[String]) -> csv::Result<Self> {
        // Find the start

        let mut tbuf: Option<DateTime<Utc>> = start;
        if tbuf.is_none() {
            for path in paths {
                let mut tweets_csv = csv::ReaderBuilder::new()
                    .from_path(&path)?;
                for tweet in tweets_csv.deserialize::<ProcessedTweetRecord>() {
                    if let Ok(tweet) = tweet.map(ProcessedTweetRecord::processed_tweet) {
                        let ts = tweet.timestamp();
                        match tbuf {
                            Some(t) => if ts < t { tbuf = Some(ts) },
                            None => tbuf = Some(ts),
                        }
                    }
                }
            }
        }


        let mut series = Self::new(tbuf.expect("No start date found!"), interval);
        
        for path in paths {
            let mut tweets_csv = csv::ReaderBuilder::new()
                .from_path(&path)?;
            for tweet in tweets_csv.deserialize::<ProcessedTweetRecord>() {
                if let Ok(tweet) = tweet.map(ProcessedTweetRecord::processed_tweet) {
                    let tw_words = tweet.tweet.processed_words(&[]);
                    let n = tw_words.iter().filter(|x| words.contains(x)).count();
                    if n > 0 {   
                        series.insert(DataPoint { time: tweet.timestamp(), val: tweet.value() });
                    }
                }
            }
        }


        Ok(series)
    }
}

impl TweetSeries<ProcessedTweet> {
    pub fn from_processed_tweets_csv<P: AsRef<std::path::Path>>(paths: &[P], start: Option<DateTime<Utc>>, interval: chrono::Duration) -> csv::Result<Self> {
        // Find the start

        let mut tbuf: Option<DateTime<Utc>> = start;
        if tbuf.is_none() {
            for path in paths {
                let mut tweets_csv = csv::ReaderBuilder::new()
                    .from_path(&path)?;
                for tweet in tweets_csv.deserialize::<ProcessedTweetRecord>() {
                    if let Ok(tweet) = tweet.map(ProcessedTweetRecord::processed_tweet) {
                        let ts = tweet.timestamp();
                        match tbuf {
                            Some(t) => if ts < t { tbuf = Some(ts) },
                            None => tbuf = Some(ts),
                        }
                    }
                }
            }
        }


        let mut series = Self::new(tbuf.expect("No start date found!"), interval);
        
        for path in paths {
            let mut tweets_csv = csv::ReaderBuilder::new()
                .from_path(&path)?;
            for tweet in tweets_csv.deserialize::<ProcessedTweetRecord>() {
                if let Ok(tweet) = tweet.map(ProcessedTweetRecord::processed_tweet) {
                    series.insert(tweet);
                }
            }
        }


        Ok(series)
    }
}

impl<T: TimeSeriesItem> TweetSeries<T> {
    pub fn new(start: DateTime<Utc>, interval: chrono::Duration) -> Self {
        Self {
            start,
            interval,
            data: Vec::new(),
        }
    }

    /// Spits the tweet back out of not inserted.
    pub fn insert(&mut self, item: T) -> Option<T> {
        let dur = item.timestamp().signed_duration_since(self.start);
        if dur.num_milliseconds() < 0 {
            return Some(item);
        }

        let idx = (dur.num_milliseconds() / self.interval.num_milliseconds()) as usize;

        if self.data.len() > idx {
            self.data[idx].push(item);
        } else {
            for _ in self.data.len()..idx {
                self.data.push(Vec::new()); // Fill in data points in between.
            }
            self.data.push(vec![item]);
        }

        None
    }

    pub fn date_for(&self, i: usize) -> DateTime<Utc> {
        self.start + (self.interval * i as i32)
    }

    pub fn start_date(&self) -> DateTime<Utc> {
        self.start
    }

    pub fn ave_series_days(&self) -> Vec<(f64, f64)> {
        self.data.iter().enumerate().map(|(i, x)| ((self.interval * i as i32).num_days() as f64, x.iter().map(TimeSeriesItem::value).sum::<f64>() / x.len() as f64)).collect()
    }

    // This only works if the interval is at least a day.
    pub fn ave_series_dt(&self) -> Vec<(DateTime<Utc>, f64)> {
        self.data.iter().enumerate().map(|(i, x)| (self.start + (self.interval * i as i32), x.iter().map(TimeSeriesItem::value).sum::<f64>() / x.len() as f64)).collect()
    }

    pub fn sum_series_days(&self) -> Vec<(f64, f64)> {
        self.data.iter().enumerate().map(|(i, x)| ((self.interval * i as i32).num_days() as f64, x.iter().map(TimeSeriesItem::value).sum::<f64>())).collect()
    }

    // This only works if the interval is at least a day.
    pub fn sum_series_dt(&self) -> Vec<(DateTime<Utc>, f64)> {
        self.data.iter().enumerate().map(|(i, x)| (self.start + (self.interval * i as i32), x.iter().map(TimeSeriesItem::value).sum::<f64>())).collect()
    }

    pub fn end_date(&self) -> DateTime<Utc> {
        self.start + (self.interval * self.data.len() as i32)
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ExportRecord {
    date: String,
    val: f64,
}

pub fn export<'a>(data: impl Iterator<Item=(DateTime<Utc>, f64)>, path: impl AsRef<std::path::Path>, date_format: &str) -> csv::Result<()> {
    let mut writer = csv::WriterBuilder::new().has_headers(true).from_path(path)?;
    for (i, (time, val)) in data.into_iter().enumerate() {
        if let Err(e) = writer.serialize(ExportRecord { date: time.format(date_format).to_string(), val }) {
            println!("Failed to serialise data point {}: {}", i, e)
        }
    }

    Ok(())
}

pub fn linear_graph<P: AsRef<std::path::Path>>(data: Vec<(DateTime<Utc>, f64)>, comparison: Option<Vec<(DateTime<Utc>, f64)>>, start: DateTime<Utc>, end: DateTime<Utc>, width: u32, height: u32, lsize: u32, stroke_width: u32, y_min: Option<f64>, y_max: Option<f64>, date_fmt: &str, caption: impl AsRef<str>, y_desc: impl Into<String>, path: P) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(&path, (width, height)).into_drawing_area();
    root.fill(&WHITE)?;

    let series = data;

    assert!(!series.is_empty());

    let mut min = series.iter().map(|(_, v)| *v).reduce(f64::min).unwrap();
    let mut max = series.iter().map(|(_, v)| *v).reduce(f64::max).unwrap();

    if let Some(v) = y_min {
        min = v;
    }

    if let Some(v) = y_max {
        max = v;
    }

    let pad = (max - min) * 0.1;

    let mut chart = ChartBuilder::on(&root)
        .margin(6 + lsize / 2)
        .caption(
            caption,
            ("sans-serif", 40),
        )
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Right, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(
            start..end,
            min - pad..max + pad,
        )?;

    chart
        .configure_mesh()
        .x_label_style(("sans-serif", lsize))
        .y_label_style(("sans-serif", lsize))
        .disable_x_mesh()
        .disable_y_mesh()
        .axis_style(ShapeStyle { color: BLACK.into(), filled: true, stroke_width: u32::max(stroke_width - 1, 1) },)
        .x_labels(4)
        .x_label_formatter(&|x| x.format(date_fmt).to_string())
        .max_light_lines(4)
        .y_desc(y_desc)
        .draw()?;

    chart.draw_series(LineSeries::new(
        series.into_iter(),
        ShapeStyle { color: BLUE.into(), filled: true, stroke_width },
    ))?;

    if let Some(comparison) = comparison {
        chart.draw_series(LineSeries::new(
            comparison.into_iter(),
            ShapeStyle { color: MAGENTA.into(), filled: true, stroke_width },
        ))?;
    
    }

    // chart.draw_series(
    //     DATA.iter()
    //         .map(|(y, m, t)| Circle::new((Utc.ymd(*y, *m, 1), *t), 3, BLUE.filled())),
    // )?;

    // To avoid the IO failure being ignored silently, we manually call the present function
    root.present()?;

    Ok(())
}