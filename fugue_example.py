import pandas as pd
from fugue import transform
from fugue_duckdb import DuckDBExecutionEngine

# Sample Data
data = pd.DataFrame({
    "id": [1, 2, 3, 4],
    "value": [10, 20, 30, 40]
})

# Define a transformation function
def add_square(df: pd.DataFrame) -> pd.DataFrame:
    df["value_squared"] = df["value"] ** 2
    return df

# Run transformation with Fugue
result = transform(
    data,
    add_square,
    schema="id:int,value:int,value_squared:int",  # required schema
    engine="pandas"   # could be "dask", "spark", "duckdb", etc.
)

print(result)
