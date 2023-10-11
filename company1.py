import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.2f" % x)


df = pd.DataFrame(pd.read_excel("company1.xlsx"))

# Display general information about the dataset
df.head()
df.columns
df.info()
df.isna().any()
df.isnull().sum()

# number of unique cities
df["SaleCityName"].nunique()
# frequencies
df["SaleCityName"].value_counts()

# number of unique Concepts
df["ConceptName"].nunique()
# frequencies
df["ConceptName"].value_counts()

# total revenue earned from sales in each city
df.groupby("SaleCityName").agg({"Price": "sum"})

# number of total revenue by each concept
df.groupby("ConceptName").agg({"Price": "sum"})

# average prices for each city
df.groupby("SaleCityName").agg({"Price": "mean"})

# average revenue by Concept type
df.groupby("ConceptName").agg({"Price": "mean"})


# average prices for each city-concept combination
df.groupby(["SaleCityName", "ConceptName"]).agg({"Price": "mean"})

# Create a new column with converted values of the SaleCheckInDayDiff variable into a categorical variable with intervals

df["EB_Score"] = pd.cut(df["SaleCheckInDayDiff"], bins=[0, 7, 30, 90, df["SaleCheckInDayDiff"].max()],
                                  labels=["LastMinuters", "PotentialPlanners", "Planners", "EarlyBookers"])

# save as new dataset
df.head(50).to_excel("eb_score.xlsx", index=False)

# Analysis of average prices and transaction count in the City-Concept-EB Score breakdown
df.groupby(["SaleCityName", "ConceptName", "EB_Score"]).agg({"Price": ["mean", "count"]})
# Analysis of average prices and transaction count in the City-Concept-Season breakdown
df.groupby(["SaleCityName", "ConceptName", "Seasons"]).agg({"Price": ["mean", "count"]})
# Analysis of average prices and transaction count in the City-Concept-CInDay breakdown
df.groupby(["SaleCityName", "ConceptName", "CInDay"]).agg({"Price": ["mean", "count"]})

# Sorted output of City-Concept-Season breakdown by price saved as agg_df
agg_df = df.groupby(["SaleCityName", "ConceptName", "Seasons"]).agg({"Price": "mean"}).sort_values("Price", ascending=False)
agg_df.head()

# Convert index to variable names
agg_df = agg_df.reset_index()
agg_df.head()

# Define and add new level-based customers (personas) as variables to the dataset
agg_df["sales_level_based"] = agg_df[["SaleCityName", "ConceptName", "Seasons"]].agg(lambda x: '_'.join(x).upper(), axis=1)
agg_df.head()

# Segment the new customers (personas) and analyze them
agg_df["Segment"] = pd.qcut(agg_df["Price"], 4, labels=["D", "C", "B", "A"])

agg_df.groupby("Segment").agg({"Price": ["mean", "max", "sum"]})

# Classify and predict how much revenue new customers can generate
agg_df.sort_values("Price")

agg_df[agg_df["sales_level_based"] == "ANTALYA_HERŞEY DAHIL_HIGH"]

agg_df[agg_df["sales_level_based"] == "GIRNE_YARIM PANSIYON_LOW"]["Segment"]
