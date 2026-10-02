"""Create a synthetic zillow.csv with the same columns as the real file (for smoke-testing the pipeline only)."""
import sys
import numpy as np
import pandas as pd


def make(n=12000, seed=0, path="synthetic_zillow.csv"):
    rng = np.random.RandomState(seed)
    dates = pd.to_datetime("2016-01-01") + pd.to_timedelta(rng.randint(0, 670, n), unit="D")
    sqft = np.exp(rng.normal(7.4, 0.4, n))
    bed = rng.randint(1, 7, n).astype(float)
    bath = np.clip(bed - rng.randint(0, 3, n) + rng.choice([0, .5], n), 1, 7)
    lat = rng.normal(34.0e6, 0.2e6, n); lon = rng.normal(-118.2e6, 0.3e6, n)
    tax = sqft * rng.normal(200, 40, n) + rng.normal(0, 2e4, n)
    yb = rng.randint(1900, 2016, n).astype(float)
    d = {
        "logerror": 0.12 * rng.standard_t(3, n) + 0.01 * (bath - 2) + 0.05 * np.sin((yb - 1900) / 18.0) + 0.04 * np.exp(-(((lat - 34.0e6) / 0.12e6) ** 2 + ((lon + 118.2e6) / 0.15e6) ** 2)),
        "transactiondate": dates.strftime("%Y-%m-%d"),
        "airconditioningtypeid": np.where(rng.rand(n) < 0.35, rng.choice([1, 13, 5], n), np.nan),
        "architecturalstyletypeid": np.where(rng.rand(n) < 0.01, 7, np.nan),
        "basementsqft": np.where(rng.rand(n) < 0.002, 300, np.nan),
        "bathroomcnt": bath, "bedroomcnt": bed,
        "buildingclasstypeid": np.where(rng.rand(n) < 0.002, 4, np.nan),
        "buildingqualitytypeid": np.where(rng.rand(n) < 0.6, rng.randint(1, 12, n), np.nan),
        "calculatedbathnbr": bath, "decktypeid": np.where(rng.rand(n) < 0.01, 66, np.nan),
        "finishedfloor1squarefeet": np.where(rng.rand(n) < 0.07, sqft * .6, np.nan),
        "calculatedfinishedsquarefeet": sqft, "finishedsquarefeet12": np.where(rng.rand(n) < 0.9, sqft, np.nan),
        "finishedsquarefeet13": np.where(rng.rand(n) < 0.003, sqft, np.nan),
        "finishedsquarefeet15": np.where(rng.rand(n) < 0.04, sqft, np.nan),
        "finishedsquarefeet50": np.where(rng.rand(n) < 0.07, sqft * .6, np.nan),
        "finishedsquarefeet6": np.where(rng.rand(n) < 0.005, sqft, np.nan),
        "fips": rng.choice([6037, 6059, 6111], n, p=[.65, .27, .08]),
        "fireplacecnt": np.where(rng.rand(n) < 0.1, rng.randint(1, 3, n), np.nan),
        "fullbathcnt": np.floor(bath), "garagecarcnt": np.where(rng.rand(n) < 0.3, rng.randint(1, 4, n), np.nan),
        "garagetotalsqft": np.where(rng.rand(n) < 0.3, rng.normal(400, 100, n), np.nan),
        "hashottuborspa": np.where(rng.rand(n) < 0.02, True, np.nan),
        "heatingorsystemtypeid": np.where(rng.rand(n) < 0.6, rng.choice([2, 7, 24, 6], n), np.nan),
        "latitude": lat, "longitude": lon,
        "lotsizesquarefeet": np.where(rng.rand(n) < 0.9, np.exp(rng.normal(8.8, 0.8, n)), np.nan),
        "poolcnt": np.where(rng.rand(n) < 0.18, 1.0, np.nan), "poolsizesum": np.where(rng.rand(n) < 0.01, 500, np.nan),
        "pooltypeid10": np.where(rng.rand(n) < 0.01, 1.0, np.nan), "pooltypeid2": np.where(rng.rand(n) < 0.015, 1.0, np.nan),
        "pooltypeid7": np.where(rng.rand(n) < 0.16, 1.0, np.nan),
        "propertycountylandusecode": rng.choice(["0100", "0101", "122", "010C", "0200", "34"], n),
        "propertylandusetypeid": rng.choice([261, 266, 246, 269, 247], n, p=[.7, .2, .04, .04, .02]),
        "propertyzoningdesc": np.where(rng.rand(n) < 0.35, rng.choice(["LAR1", "LARS", "SCUR2", "LAR3", "PSR6"], n), None),
        "rawcensustractandblock": 60371066.461001 + rng.randint(0, 50, n),
        "regionidcity": np.where(rng.rand(n) < 0.98, rng.randint(5000, 5040, n), np.nan),
        "regionidcounty": rng.choice([3101, 1286, 2061], n, p=[.65, .27, .08]),
        "regionidneighborhood": np.where(rng.rand(n) < 0.4, rng.randint(100, 160, n), np.nan),
        "regionidzip": rng.randint(96000, 96060, n).astype(float),
        "roomcnt": np.where(rng.rand(n) < 0.6, 0, bed + bath + 2), "storytypeid": np.where(rng.rand(n) < 0.002, 7, np.nan),
        "threequarterbathnbr": np.where(rng.rand(n) < 0.1, 1, np.nan), "typeconstructiontypeid": np.where(rng.rand(n) < 0.003, 6, np.nan),
        "unitcnt": np.where(rng.rand(n) < 0.67, 1, np.nan), "yardbuildingsqft17": np.where(rng.rand(n) < 0.03, 300, np.nan),
        "yardbuildingsqft26": np.where(rng.rand(n) < 0.001, 100, np.nan), "yearbuilt": yb,
        "numberofstories": np.where(rng.rand(n) < 0.25, rng.randint(1, 3, n), np.nan),
        "fireplaceflag": np.where(rng.rand(n) < 0.002, True, np.nan),
        "structuretaxvaluedollarcnt": tax * 0.6, "taxvaluedollarcnt": tax, "assessmentyear": 2015.0,
        "landtaxvaluedollarcnt": tax * 0.4, "taxamount": tax * 0.012,
        "taxdelinquencyflag": np.where(rng.rand(n) < 0.02, "Y", None),
        "taxdelinquencyyear": np.where(rng.rand(n) < 0.02, 14.0, np.nan),
        "censustractandblock": 60371066461001 + rng.randint(0, 50, n) * 1000,
    }
    pd.DataFrame(d).to_csv(path, index=False)
    print("wrote", path, n)


if __name__ == "__main__":
    make(int(sys.argv[1]) if len(sys.argv) > 1 else 12000, path=sys.argv[2] if len(sys.argv) > 2 else "synthetic_zillow.csv")
