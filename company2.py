import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.2f" % x)
df = pd.read_csv("company2.csv")

# Display general information about the dataset
df.head()
df.info()
df.isnull().sum()
df.describe().T

# Analyze unique values and frequencies of variables
df["SOURCE"].value_counts()

df["PRICE"].nunique()
df["PRICE"].value_counts()

df["COUNTRY"].value_counts()

# Analyze the variables with aggregate functions

df.groupby("COUNTRY").agg({"PRICE": "sum"})

df.groupby("SOURCE").agg({"PRICE": "sum"})

df.groupby("COUNTRY").agg({"PRICE": "mean"})

df.groupby(["COUNTRY", "SOURCE"]).agg({"PRICE": "mean"})

# Sorted output of mutliple variables' breakdown by price saved as agg_df
agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"})\
    .sort_values("PRICE", ascending=False)

agg_df.head()

# Convert index to variable names
agg_df.reset_index(inplace=True)


# Create new categorical column with age variable, divided by with intervals
agg_df["AGE"].describe()

agg_df["AGE_CAT"] = pd.cut(x=agg_df["AGE"], bins=[0, 18, 23, 30, 40, 66],
                           labels=["0-18", "19-23", "24-30", "31-40", "41-66"])

# Define level-based customers (personas) as a new column
# apply function is used to apply a lambda function to each row of the selected columns
# join function is used to concatenate the values from the selected columns with underscores ('_')

agg_df["customers_level_based"] = agg_df[["COUNTRY", "SOURCE", "SEX", "AGE_CAT"]]\
    .apply(lambda x:"_".join(x).upper(), axis=1)

agg_df.head()

# Segment the personas by price and analyze them

agg_df2 = agg_df[["customers_level_based", "PRICE"]]
agg_df2.head()

agg_df2 = agg_df2.groupby("customers_level_based").agg({"PRICE": "mean"})

agg_df2 = agg_df2.reset_index()
agg_df2["customers_level_based"].value_counts()
agg_df2.head()

agg_df2["SEGMENT"] = pd.qcut(agg_df2["PRICE"], 4, labels=["D", "C", "B", "A"])
agg_df2.head()

agg_df2.groupby("SEGMENT").agg({"PRICE": ["mean", "max", "sum"]})
 

# Classify and predict how much revenue new customers can generate

agg_df2[agg_df2["customers_level_based"] == "TUR_ANDROID_FEMALE_31-40"][["PRICE", "SEGMENT"]]

agg_df2[agg_df2["customers_level_based"] == "FRA_IOS_FEMALE_31-40"][["PRICE", "SEGMENT"]]
