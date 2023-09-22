use std::borrow::Cow;
use std::fs::OpenOptions;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::SystemTime;
use std::error::Error;
use indicatif::ProgressBar;
use nitscrape::net::{TorClientManager, TorKernelSettings};
use nitscrape::twt::{NitScrapeError, TweetError, Tweet};
use nitscrape::{twt, table};
use nitscrape::net;
use nitscrape::table::{TweetEntry, TweetCsvReader, CsvLayout};
use tokio_util::sync::CancellationToken;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Resume {
    i: usize,
    output_path: String,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Settings {
    pub input_path: String, 
    pub layout: CsvLayout, 
    pub output_path: String, 
    pub tor_path: String, 
    pub base_config: String, 
    pub dump_path: String,
    pub resume_path: String,
    pub num_kernels: u32, 
    pub sample_skip: usize,
    pub timeout: f64,
    pub kernel_settings: TorKernelSettings,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            input_path: "data/skel.csv".to_owned(),
            layout: CsvLayout::default(),
            output_path: "data/hydr.csv".to_owned(),
            tor_path: "/usr/bin/tor".to_owned(),
            base_config: "/torrc_base".to_owned(),
            dump_path: "data/dump.csv".to_owned(),
            resume_path: "data/resume.json".to_owned(),
            num_kernels: 100,
            sample_skip: 0,
            timeout: 30.0,
            kernel_settings: Default::default(),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ResumeMethod {
    ResumeFile,
    TweetId,
    None,
}

pub async fn begin_tor_farm(settings: Settings, resume: ResumeMethod, progress: bool) -> Result<(), Box<dyn Error>> {
    assert!(settings.num_kernels < 3000);
    // First we need to set up the network manager.
    println!("Establishing {} tor clients (and circuits)...", settings.num_kernels);
    let tor = TorClientManager::generate_configs(settings.tor_path, settings.base_config, 7000..7000+settings.num_kernels, settings.timeout).await?; // We have a port per kernel.
    
    println!("Established {} out of {} clients.", tor.client_count(), settings.num_kernels);
    // Task manager
    println!("Creating load manager...");
    let mut farm: net::AsyncLoadManager<net::TorClient, TweetEntry> = net::AsyncLoadManager::new(500);

    println!("Executing tor kernels...");
    let _ = farm.execute_tor_kernels(TorKernelSettings::default(), tor);

    if farm.kernel_count() == 0 {
        return Err("No tor instances available!".into());
    }

    let pb = {
        if progress {
            // We need to read the length of the input:
            println!("Reading input csv file for progress bar (this may take a while)...");
            Some(ProgressBar::new(TweetCsvReader::read_csv(&settings.input_path, settings.layout.clone())?.count() as u64))
        } else {
            None
        }
    };


    // Start feeder:
    println!("Loading csv reader and writer...");
    let mut csv = TweetCsvReader::read_csv(&settings.input_path, settings.layout)?;

    let mut i: usize = 0;

    let dir = std::path::Path::new(&settings.output_path).parent().map(|x| x.to_string_lossy()).unwrap_or(Cow::Borrowed(""));
    let stem = std::path::Path::new(&settings.output_path).file_stem().expect("No file stem!").to_string_lossy();
    let ext = std::path::Path::new(&settings.output_path).extension().map(|x| x.to_string_lossy());
    let mut output_path = settings.output_path.clone();

    if resume == ResumeMethod::ResumeFile {
        println!("Checking for resume file at {}...", &settings.resume_path);
        if let Ok(res_str) = std::fs::read_to_string(&settings.resume_path) {
            match serde_json::from_str::<Resume>(&res_str) {
                Ok(resume) => {
                    if &resume.output_path == &settings.output_path {
                        println!("Resuming from resume.json at index {}", resume.i);
                        i = resume.i;

                        // Bring iterator up to position:
                        if i > 0 {
                            let _ = csv.tweet_entries().nth(i - 1);
                        }
        
                    } else {
                        println!("Settings output and resume output do not match - not resuming...")
                    }
                },
                Err(e) => println!("Resume.json deserialisation error: {}", e),
            }
        } else {
            println!("No resume.json file available.")
        }
    
        if let Ok(dumped) = table::read_dump(&settings.dump_path) {
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
    } else if resume == ResumeMethod::TweetId {
        // Check output file for tweets.
        let mut output = csv::ReaderBuilder::new().from_path(&settings.output_path)?;
        let mut buf: Option<Tweet> = None;
        for tweet in output.deserialize::<Tweet>() {
            if let Ok(tweet) = tweet {
                buf = Some(tweet);
            }
        }

        if let Some(tw) = buf {
            println!("Attemtping to resume at tweet id {}...", tw.id);
            for j in 0.. {
                if let Some(Ok(t)) = csv.tweet_entries().next() {
                    if t.id == tw.id {
                        i = j;
                        println!("Found tweet at index {}.", j);
                        break;
                    }
                }
            }
        } else {
            println!("No valid tweets found in output file - no resume will occur.");
        }
    }

    let _ = std::fs::remove_file(&settings.dump_path);

    if i == 0 { // We're starting from the beginning so we don't want to overwrite existing stuff. Also if we are resuming from tweets and the tweet id isnt found then this is executed.
        for j in 1.. {
            match std::fs::read_to_string(&output_path) {
                Ok(s) => {
                    if s.is_empty() {
                        break;
                    } else {
                        let mut pb = PathBuf::new();
                        if !dir.is_empty() {
                            pb.push(dir.as_ref());
                        }
                        let mut f = format!("{}_{}", stem, j);
                        if let Some(ext) = &ext {
                            f += ".";
                            f += &ext;
                        }
                        pb.push(f);
                        output_path = pb.to_string_lossy().into();
                    }
                },
                Err(e) => match e.kind() {
                    std::io::ErrorKind::NotFound => break, // proceed with path
                    _ => println!("IO error when trying to validate file: {}", e),
                },
                
            }
        }
    }

    println!("Loading output writer with path: {}", &output_path);

    let output_csv = OpenOptions::new()
        .write(true)
        .append(resume != ResumeMethod::None)
        .create(true)
        .open(&output_path)?;

    let mut writer = csv::Writer::from_writer(output_csv);

    let ct = CancellationToken::new();

    // Spawn input listener
    println!("Starting input listener... Press 'k' to end the process and save progress in resume.json");
    let iohandle = {
        let ct = ct.clone();
        tokio::task::spawn_blocking(move || {
            while !ct.is_cancelled() {
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
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            
        })
    };

    println!("Beginning write loop with sample skip {}...", settings.sample_skip);
    let cur_index: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(i));
    let sender_h = {
        let ct: CancellationToken = ct.clone();
        let is = farm.input_sender().clone();
        let cur_index = cur_index.clone();
        tokio::spawn(async move {
            while !ct.is_cancelled() {
                if is.capacity() > 0 {
                    match csv.tweet_entries().nth(settings.sample_skip) {
                        Some(Ok(next_entry)) => {
                            // Add to queue.
                            if let Err(e) = is.send(
                                twt::load_request(next_entry)
                            ).await {
                                println!("Failed to send input: {}", e);
                            }
                        },
                        None => {
                            println!("No tweets left - exiting!");
                            ct.cancel();
                            break;
                        },
                        Some(Err(e)) => println!("Failed to load csv: {}", e),
                    }
                    cur_index.fetch_add(1 + settings.sample_skip, Ordering::Relaxed);
                }
            }
        })
    };

    let mut t0: Option<SystemTime> = None;
    let mut max_i: u64 = i as u64;
    let mut write_count: u64 = 0;
    let mut req_count: u64 = 0;
    let mut resp_count: u64 = 0;
    while !ct.is_cancelled() {
        // Iterate through outputs.
        for resp in farm.try_responses() {
            req_count += resp.tries;
            resp_count += 1;
            if resp.response.status().is_success() {
                if let Ok(html) = resp.response.text().await {
                    match twt::parse_nitter(resp.req_data.id, html) {
                        Ok(tweet) =>  {
                            let _ = writer.serialize(tweet);

                            // Write to progress bar
                            if let Some(pb) = &pb {
                                if let Some(p) = resp.req_data.pos {
                                    if p > max_i {
                                        max_i = p;
                                        pb.set_position(max_i as u64);
                                    }
                                }
                            }

                            write_count += 1;
                            if write_count % 1000 == 0 {
                                let ci = cur_index.load(Ordering::Relaxed);

                                let mut rw: f64 = f64::NAN;
                                let mut ri: f64 = f64::NAN;

                                let ar: f64 = req_count as f64 / resp_count as f64;

                                if let Some(t0) = t0 {
                                    let delta = SystemTime::now().duration_since(t0).unwrap();
                                    rw = write_count as f64 / delta.as_secs_f64();
                                    ri = (ci - i) as f64 / delta.as_secs_f64();
                                } else {
                                    t0 = Some(SystemTime::now());
                                }
                                if let Some(pb) = &pb {
                                    pb.println(format!("Written {} tweets (i={}, rw={}, ri={}, ar={})", write_count, ci, rw, ri, ar))
                                } else {
                                    println!("Written {} tweets (i={}, rw={}, ri={}, ar={})", write_count, ci, rw, ri, ar);
                                }
                                
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
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    println!("Awaiting sender process...");
    sender_h.await?;

    println!("Exited main loop, sending kill instruction to kernels...");
    farm.send_kernel_kill();

    let new_idx = cur_index.load(Ordering::Relaxed);

    let resume = Resume { i: new_idx, output_path: output_path };
    if let Ok(file) = OpenOptions::new().create(true).write(true).open(&settings.resume_path) {
        match serde_json::to_writer(file, &resume) {
            Ok(_) => println!("Successfully wrote resume.json with i={}", i),
            Err(e) => println!("Failed to write resume.json: {}", e),
        }
    }
    

    println!("Awaiting kernel exit and dump...");
    let dump = farm.dump().await;
    match table::write_dump(&settings.dump_path, &dump) {
        Ok(_) => println!("Successfully wrote dump file with {} items", dump.len()),
        Err(e) => println!("Failed to write dump: {}", e),
    }

    println!("Stopping io process...");
    ct.cancel();
    let _ = iohandle.await;

    println!("Cleanup completed successfully.");
    Ok(())
}
