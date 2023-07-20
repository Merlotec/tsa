use std::fs::OpenOptions;
use std::{fmt, error::Error, fs::File};
use nitscrape::net::{TorClientManager, TorKernelSettings};
use nitscrape::twt::{NitScrapeError, TweetError};
use nitscrape::{twt, table};
use nitscrape::net;
use nitscrape::table::{TweetEntry, TweetCsvReader, CsvLayout};
use reqwest::header::HeaderValue;
use reqwest::{Method, Url};
use tokio_util::sync::CancellationToken;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Resume {
    i: usize,
    output_path: String,
}

#[derive(Default, serde::Serialize, serde::Deserialize)]
pub struct Settings {
    pub input_path: String, 
    pub layout: CsvLayout, 
    pub output_path: String, 
    pub tor_path: String, 
    pub base_config: String, 
    pub num_kernels: u32, 
    pub sample_skip: usize,
}

pub async fn begin_tor_farm(settings: Settings, resume: bool) -> Result<(), Box<dyn Error>> {
    assert!(settings.num_kernels < 3000);
    // First we need to set up the network manager.
    println!("Establishing {} tor clients (and circuits)...", settings.num_kernels);
    let tor = TorClientManager::generate_configs(settings.tor_path, settings.base_config, 7000..7000+settings.num_kernels).await?; // We have a port per kernel.
    
    // Task manager
    println!("Creating load manager...");
    let mut farm: net::AsyncLoadManager<net::TorClient, TweetEntry> = net::AsyncLoadManager::new(500);

    println!("Executing tor kernels...");
    let _ = farm.execute_tor_kernels(TorKernelSettings::default(), tor);

    // Start feeder:
    println!("Loading csv reader and writer...");
    let mut csv = TweetCsvReader::read_csv(settings.input_path, settings.layout)?;

    // Detect current position of tweets:
    

    let input_csv = OpenOptions::new()
        .write(true)
        .append(true)
        .create(true)
        .open(&settings.output_path)?;

    let mut i: usize = 0;

    if resume {
        println!("Checking for resume.json...");
        if let Ok(file) = OpenOptions::new().open("resume.json") {
            if let Ok(resume) = serde_json::from_reader::<_, Resume>(file) {
                if &resume.output_path == &settings.output_path {
                    println!("Resuming from resume.json at index {}", resume.i);
                    i = resume.i;
                } else {
                    println!("Settings output and resume output do not match - not resuming...")
                }
            }
        }
    
        if let Ok(dumped) = table::read_dump("dump.csv") {
            println!("Queuing dumped items...");
            for entry in dumped {
                if let Err(e) = farm.input_sender().send(
                    twt::load_request(entry)
                ).await {
                    println!("Failed to send dumped input: {}", e);
                }
            }
        } else {
            println!("No dump file available.");
        }
    }

    let _ = std::fs::remove_file("dumped.csv");

    let mut writer = csv::Writer::from_writer(input_csv);

    // Bring iterator up to position:
    if i > 0 {
        let _ = csv.tweet_entries().nth(i - 1);
    }

    let ct = CancellationToken::new();

    // Spawn input listener
    println!("Starting input listener... Press 'k' to end the process and save progress in resume.json");
    tokio::task::spawn_blocking(move || {
        loop {
            use std::io::{stdin, stdout, Write};
            let mut s = String::new();
            let _ = stdout().flush();
            stdin()
                .read_line(&mut s)
                .expect("Did not enter a correct string");
            if let Some('\n') = s.chars().next_back() {
                s.pop();
            }
            if let Some('\r') = s.chars().next_back() {
                s.pop();
            }

            if s == "k" {
                println!("Sending cancellation token due to input...");
                ct.cancel();
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        
    });

    println!("Beginning write loop with sample skip {}...", settings.sample_skip);
    let mut write_count: usize = 0;

    loop {
        if farm.input_sender().capacity() > 0 {
            match csv.tweet_entries().nth(settings.sample_skip) {
                Some(Ok(next_entry)) => {
                    // Add to queue.
                    if let Err(e) = farm.input_sender().send(
                        twt::load_request(next_entry)
                    ).await {
                        println!("Failed to send input: {}", e);
                    }
                },
                None => break,
                Some(Err(e)) => println!("Failed to load csv: {}", e),
            }
            i += 1 + settings.sample_skip;
        }

        // Iterate through outputs.
        for resp in farm.output() {
            if resp.response.status().is_success() {
                if let Ok(html) = resp.response.text().await {
                    match twt::parse_nitter(resp.req_data.id, html) {
                        Ok(tweet) =>  {
                            let _ = writer.serialize(tweet);
                            write_count += 1;
                            if write_count % 100 == 0 {
                                let rate = farm.ave_rate();
                                println!("Written {} tweets with rate {}", write_count, rate);
                            }
                        },
                        Err(TweetError { scrape_error: NitScrapeError::ScraperParseError, tweet_id }) => println!("Failed to parse tweet {}: Unexpected parse error", tweet_id),
                        #[cfg(debug)]
                        Err(e) => println!("Failed to parse tweet {}: {}", resp.req_data.id, e),
                        #[cfg(not(debug))]
                        Err(_) => {},
                    } 
                    
                }
            }
        }
    }

    println!("Exited main loop.");

    let resume = Resume { i, output_path: settings.output_path };
    if let Ok(file) = OpenOptions::new().create(true).write(true).open("resume.json") {
        match serde_json::to_writer(file, &resume) {
            Ok(_) => println!("Successfully wrote resume.json with i={}", i),
            Err(e) => println!("Failed to write resume.json: {}", e),
        }
    }
    

    let dump = farm.dump().await;
    match table::write_dump("dumped.csv", &dump) {
        Ok(_) => println!("Successfully wrote dump file with {} items", dump.len()),
        Err(e) => println!("Failed to write dump: {}", e),
    }
    Ok(())
}
