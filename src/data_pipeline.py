import fastf1
import time
import pandas as pd
import os
from fastf1._api import SessionNotAvailableError

# ---- Config ----
cache_dir = "data/fastf1_cache"
os.makedirs(cache_dir, exist_ok=True)
os.makedirs("data", exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)
fastf1.set_log_level('WARNING')

PROGRESS_FILE = "data/fetch_progress.csv"
RESULTS_FILE = "data/f1_all_results.parquet"
LAPS_FILE = "data/f1_all_laps.parquet"
WEATHER_FILE = "data/f1_all_weather.parquet"

YEARS = list(range(2021, 2027))
SESSION_TYPES = ['FP1', 'FP2', 'FP3', 'Q', 'SQ', 'S', 'R']
RATE_LIMIT_WAIT = 3600


print("I am running the updated file now ")

def load_progress():
    """Load set of already-fetched (year, round, session_type) tuples
    from both the progress file AND existing parquet files."""
    completed = set()

    # From progress tracker
    if os.path.exists(PROGRESS_FILE):
        df = pd.read_csv(PROGRESS_FILE)
        completed.update(zip(df['year'], df['round'], df['session_type']))

    # From existing parquet files — if data is already saved, skip it
    for filepath in [RESULTS_FILE, LAPS_FILE, WEATHER_FILE]:
        if os.path.exists(filepath):
            df = pd.read_parquet(filepath, columns=['Season', 'Round', 'SessionType'])
            completed.update(zip(df['Season'], df['Round'], df['SessionType']))

    return completed


def save_progress(year, round_no, stype):
    row = pd.DataFrame([{'year': year, 'round': round_no, 'session_type': stype}])
    row.to_csv(PROGRESS_FILE, mode='a', header=not os.path.exists(PROGRESS_FILE), index=False)


def is_rate_limited(exception):
    msg = str(exception).lower()
    return any(term in msg for term in [
        '429', 'rate limit', 'too many requests', 'calls/h',
    ])


def load_session_with_retry(year, round_no, session_type='R', retries=3, delay=5):
    for attempt in range(retries):
        try:
            session = fastf1.get_session(year, round_no, session_type)
            # FastF1 checks its own cache internally — if cached, no API call
            session.load(laps=True, telemetry=False, weather=True, messages=False)
            return session
        except Exception as e:
            if session_does_not_exist(e):
                print("Session Does not exist")
                return None
            if is_rate_limited(e):
                print(f"\n*** Rate limit hit at {year} R{round_no} {session_type}. "
                      f"Waiting {RATE_LIMIT_WAIT // 60} minutes... ***\n")
                time.sleep(RATE_LIMIT_WAIT)
                continue
            print(f"Attempt {attempt+1} failed for {year} R{round_no} {session_type}: {e}")
            time.sleep(delay)
    print(f"Failed to load {year} R{round_no} {session_type} after {retries} retries.")
    return None


def append_to_parquet(df, filepath):
    if os.path.exists(filepath):
        existing = pd.read_parquet(filepath)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(filepath, index=False)

def session_does_not_exist(exception):
    msg = str(exception).lower()
    return 'does not exist' in msg or 'no session' in msg

# ---- Main fetch loop ----
completed = load_progress()
print(f"Already have {len(completed)} sessions saved. Skipping those.\n")

SESSION_OFFSET = {          # days after Session1 (Friday/day 1)
    'FP1': 0, 'FP2': 0, 'FP3': 1,
    'Q': 1, 'SQ': 0, 'S': 1, 'R': 2,
}

now = pd.Timestamp.now(tz='UTC')

for year in YEARS:
    try:
        schedule = fastf1.get_event_schedule(year)
    except Exception as e:
        print(f"Failed to load schedule for {year}: {e}")
        continue

    for _, event in schedule.iterrows():
        round_no = event['RoundNumber']
        if round_no == 0:
            continue

        day1 = event.get('Session1DateUtc')
        location = event['Location']
        event_name = event['EventName']

        event_format = str(event.get('EventFormat', '')).lower()
        has_sprint = 'sprint' in event_format


        for stype in SESSION_TYPES:

            if (year, round_no, stype) in completed:
                print(f"Data for {year} R{round_no} {stype} already fetched. Skipping.")
                continue

            if stype in ('S', 'SQ') and not has_sprint:
                print(f"Skipping {year} R{round_no} {stype} (no sprint)")
                continue

            if stype in ('FP2', 'FP3') and has_sprint:
                print(f"Skipping {year} R{round_no} {stype} (sprint format)")
                continue

            session = load_session_with_retry(year, round_no, session_type=stype)
            if session is None:
                continue

            # skip sessions whose day hasn't arrived yet; Assumes Race on Sunday structure, change for 
            # different structures.To change, adjust SESSION_OFFSET above to reflect the number of days after Session1 that each session occurs.
            if pd.notna(day1):
                sdate = pd.Timestamp(day1) + pd.Timedelta(days=SESSION_OFFSET[stype])
                if sdate.tz is None:
                    sdate = sdate.tz_localize('UTC')
                if sdate > now:
                    continue


            metadata = {
                'Season': year,
                'Round': round_no,
                'Location': location,
                'Circuit': event_name,
                'SessionType': stype,
            }

# ---- Results ----
            try:
                results = session.results
                if results is not None and not results.empty:
                    results = results.copy()
                    for k, v in metadata.items():
                        results[k] = v
                    append_to_parquet(results, RESULTS_FILE)
            except Exception:
                pass

            # ---- Laps ----
            try:
                laps = session.laps
                if laps is not None and not laps.empty:
                    laps = laps.copy()
                    for k, v in metadata.items():
                        laps[k] = v
                    append_to_parquet(laps, LAPS_FILE)
            except Exception:
                pass

            # ---- Weather ----
            try:
                weather = session.weather_data
                if weather is not None and not weather.empty:
                    weather = weather.copy()
                    for k, v in metadata.items():
                        weather[k] = v
                    append_to_parquet(weather, WEATHER_FILE)
            except Exception:
                pass

            save_progress(year, round_no, stype)
            print(f"Loaded {year} {event_name} - {stype}")
            time.sleep(0.2)

print("\n=== Fetch complete ===")
for f in [RESULTS_FILE, LAPS_FILE, WEATHER_FILE]:
    if os.path.exists(f):
        df = pd.read_parquet(f)
        print(f"{f}: {len(df)} rows, {len(df.columns)} columns")