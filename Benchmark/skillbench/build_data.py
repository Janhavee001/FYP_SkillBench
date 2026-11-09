"""
Download and prepare every SkillBench dataset, reproducibly. Raw downloads are cached in data/raw/;
prepared tables go to data/<dataset>/. Run once:

    python build_data.py

Preparation is deliberately minimal: files are converted to CSV/Parquet where needed so every system
reads the same bytes, but the data are NOT cleaned. Missing-value codes, cancellations, negative
amounts, orphans and outliers are left in, because handling them is part of what is being tested.
"""
from __future__ import annotations

import io
import json
import pathlib
import shutil
import sys
import urllib.request
import zipfile

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
DATA = ROOT / "data"

SOURCES = {
    "diabetes.zip": "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip",
    "online_retail_ii.zip": "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
    "bike_sharing.zip": "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
    "beijing_pm25.zip": "https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip",
    "yellow_tripdata_2024-01.parquet": "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet",
    "ml-1m.zip": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "f1db-csv.zip": "https://github.com/f1db/f1db/releases/latest/download/f1db-csv.zip",
}
OLIST = ["olist_customers_dataset.csv", "olist_geolocation_dataset.csv", "olist_order_items_dataset.csv",
         "olist_order_payments_dataset.csv", "olist_order_reviews_dataset.csv", "olist_orders_dataset.csv",
         "olist_products_dataset.csv", "olist_sellers_dataset.csv", "product_category_name_translation.csv"]
OLIST_URL = "https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets/{}"


def fetch(name: str, url: str) -> pathlib.Path:
    """Download with retries to a .part file, renamed only when complete, so an interrupted
    transfer is never mistaken for a finished one."""
    import subprocess
    path = RAW / name
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    print(f"  downloading {name} ...", flush=True)
    subprocess.run(["curl", "-sSL", "--fail", "--retry", "8", "--retry-delay", "3", "--retry-all-errors",
                    "-C", "-", "-o", str(part), url], check=True)
    part.replace(path)
    return path


V2_SOURCES = {
    "taxi_zone_lookup.csv": "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv",
    "imdb/title.basics.tsv.gz": "https://datasets.imdbws.com/title.basics.tsv.gz",
    "imdb/title.ratings.tsv.gz": "https://datasets.imdbws.com/title.ratings.tsv.gz",
    "imdb/title.crew.tsv.gz": "https://datasets.imdbws.com/title.crew.tsv.gz",
    "imdb/name.basics.tsv.gz": "https://datasets.imdbws.com/name.basics.tsv.gz",
    "citibike_202401.zip": "https://s3.amazonaws.com/tripdata/202401-citibike-tripdata.zip",
    **{f"flights_2024_{m}.zip": "https://transtats.bts.gov/PREZIP/"
       f"On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2024_{m}.zip" for m in (1, 2, 3)},
    "L_AIRPORT_ID.csv": "https://transtats.bts.gov/Download_Lookup.asp?Y11x72=Y_NVecbeg_VQ",
    "L_UNIQUE_CARRIERS.csv": "https://transtats.bts.gov/Download_Lookup.asp?Y11x72=Y_haVdhR_PNeeVRef",
}


def build_v2_real() -> None:
    """Large real datasets added in v2. Filtering keeps them within laptop memory while staying in the
    millions of rows: IMDb is restricted to feature films (titleType == 'movie') and the people credited on
    them; flights keep the analysis columns of the BTS file (all rows)."""
    for name, url in V2_SOURCES.items():
        fetch(name, url)
    out = lambda name: (DATA / name).mkdir(parents=True, exist_ok=True) or DATA / name  # noqa: E731

    shutil.copy(RAW / "taxi_zone_lookup.csv", out("nyc_taxi") / "taxi_zone_lookup.csv")

    imdb = out("imdb")
    if not (imdb / "movies.parquet").exists():
        print("  filtering IMDb to feature films ...", flush=True)
        basics = pd.read_csv(RAW / "imdb/title.basics.tsv.gz", sep="\t", na_values=[], keep_default_na=False,
                             quoting=3, dtype=str, usecols=["tconst", "titleType", "primaryTitle", "isAdult",
                                                            "startYear", "runtimeMinutes", "genres"])
        movies = basics[basics.titleType == "movie"].drop(columns="titleType")
        movies.to_parquet(imdb / "movies.parquet", index=False)        # '\N' kept: IMDb's missing-value code
        ids = set(movies.tconst)
        ratings = pd.read_csv(RAW / "imdb/title.ratings.tsv.gz", sep="\t", dtype={"tconst": str})
        ratings[ratings.tconst.isin(ids)].to_parquet(imdb / "ratings.parquet", index=False)
        crew = pd.read_csv(RAW / "imdb/title.crew.tsv.gz", sep="\t", dtype=str, na_values=[], keep_default_na=False)
        crew = crew[crew.tconst.isin(ids)]
        crew.to_parquet(imdb / "crew.parquet", index=False)
        directors = {d for v in crew.directors if v != "\\N" for d in v.split(",")}
        people = pd.read_csv(RAW / "imdb/name.basics.tsv.gz", sep="\t", dtype=str, na_values=[], keep_default_na=False,
                             quoting=3, usecols=["nconst", "primaryName", "birthYear", "deathYear", "primaryProfession"])
        people[people.nconst.isin(directors)].to_parquet(imdb / "directors.parquet", index=False)

    cb = out("citibike")
    if not (cb / "citibike_202401.parquet").exists():
        print("  combining Citi Bike January 2024 ...", flush=True)
        with zipfile.ZipFile(RAW / "citibike_202401.zip") as z:
            parts = [pd.read_csv(z.open(n), dtype={"start_station_id": str, "end_station_id": str})
                     for n in z.namelist() if n.endswith(".csv") and "__MACOSX" not in n]
        pd.concat(parts, ignore_index=True).to_parquet(cb / "citibike_202401.parquet", index=False)

    fl = out("flights")
    if not (fl / "flights_2024q1.parquet").exists():
        print("  combining BTS flights, January-March 2024 ...", flush=True)
        keep = ["FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", "Tail_Number",
                "OriginAirportID", "Origin", "OriginState", "DestAirportID", "Dest", "DestState", "CRSDepTime",
                "DepTime", "DepDelay", "DepDel15", "CRSArrTime", "ArrTime", "ArrDelay", "ArrDel15", "Cancelled",
                "CancellationCode", "Diverted", "CRSElapsedTime", "ActualElapsedTime", "AirTime", "Distance",
                "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]
        frames = []
        for m in (1, 2, 3):
            with zipfile.ZipFile(RAW / f"flights_2024_{m}.zip") as z:
                frames.append(pd.read_csv(z.open(zip_member(z, ".csv")), usecols=keep, low_memory=False))
        pd.concat(frames, ignore_index=True).to_parquet(fl / "flights_2024q1.parquet", index=False)
        shutil.copy(RAW / "L_AIRPORT_ID.csv", fl / "airports.csv")
        shutil.copy(RAW / "L_UNIQUE_CARRIERS.csv", fl / "carriers.csv")


def zip_member(z: zipfile.ZipFile, suffix: str) -> str:
    return next(n for n in z.namelist() if n.endswith(suffix))


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        fetch(name, url)
    for f in OLIST:
        fetch(f"olist/{f}", OLIST_URL.format(f))

    out = lambda name: (DATA / name).mkdir(parents=True, exist_ok=True) or DATA / name  # noqa: E731

    # --- single-table -------------------------------------------------------
    with zipfile.ZipFile(RAW / "diabetes.zip") as z:
        inner = z.namelist()
        if any(n.endswith(".zip") for n in inner):  # UCI wraps it twice
            z = zipfile.ZipFile(io.BytesIO(z.read(zip_member(z, ".zip"))))
        (out("diabetes") / "diabetic_data.csv").write_bytes(z.read(zip_member(z, "diabetic_data.csv")))
        (out("diabetes") / "IDS_mapping.csv").write_bytes(z.read(zip_member(z, "IDS_mapping.csv")))

    retail_csv = out("online_retail") / "online_retail_ii.csv"
    if not retail_csv.exists():
        print("  converting Online Retail II (two Excel sheets, ~1M rows; takes a few minutes) ...", flush=True)
        with zipfile.ZipFile(RAW / "online_retail_ii.zip") as z:
            xl = io.BytesIO(z.read(zip_member(z, ".xlsx")))
        sheets = pd.read_excel(xl, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
        pd.concat(sheets.values(), ignore_index=True).to_csv(retail_csv, index=False)

    with zipfile.ZipFile(RAW / "bike_sharing.zip") as z:
        (out("bike_sharing") / "hour.csv").write_bytes(z.read(zip_member(z, "hour.csv")))

    with zipfile.ZipFile(RAW / "beijing_pm25.zip") as z:
        (out("beijing_pm25") / "beijing_pm25.csv").write_bytes(z.read(zip_member(z, ".csv")))

    shutil.copy(RAW / "yellow_tripdata_2024-01.parquet", out("nyc_taxi") / "yellow_tripdata_2024-01.parquet")

    # --- multi-table --------------------------------------------------------
    for f in OLIST:
        shutil.copy(RAW / "olist" / f, out("olist") / f)

    ml = out("movielens")
    with zipfile.ZipFile(RAW / "ml-1m.zip") as z:
        spec = {"users.dat": ["user_id", "gender", "age", "occupation", "zip_code"],
                "movies.dat": ["movie_id", "title", "genres"],
                "ratings.dat": ["user_id", "movie_id", "rating", "timestamp"]}
        for member, cols in spec.items():
            raw = z.read(zip_member(z, member)).decode("latin-1")
            rows = [line.split("::") for line in raw.splitlines() if line]
            pd.DataFrame(rows, columns=cols).to_csv(ml / member.replace(".dat", ".csv"), index=False)

    f1 = out("f1db")
    with zipfile.ZipFile(RAW / "f1db-csv.zip") as z:
        for n in z.namelist():
            if n.endswith(".csv"):
                (f1 / pathlib.Path(n).name.replace("f1db-", "").replace("-", "_")).write_bytes(z.read(n))

    sys.path.insert(0, str(ROOT.parent.parent / "Skills"))
    import importlib.util
    spec_ = importlib.util.spec_from_file_location("shop", ROOT.parent.parent / "Skills" / "demo" / "make_shop_tables.py")
    shop = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(shop)
    for name, df in shop.make_shop().items():
        df.to_csv(out("shop") / f"{name}.csv", index=False)

    build_v2_real()

    summary = {}
    for d in sorted(p for p in DATA.iterdir() if p.is_dir() and p.name != "raw"):
        files = sorted(f for f in d.iterdir() if f.suffix in (".csv", ".parquet"))
        summary[d.name] = {f.name: round(f.stat().st_size / 1e6, 1) for f in files}
    (DATA / "manifest.json").write_text(json.dumps(summary, indent=1))
    for k, v in summary.items():
        print(f"{k}: {len(v)} file(s), {sum(v.values()):.1f} MB")


if __name__ == "__main__":
    main()
