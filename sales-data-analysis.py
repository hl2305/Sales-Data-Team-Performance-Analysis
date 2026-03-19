# Data Preparation 

## Importing libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sklearn
import matplotlib.cbook as cbook
from matplotlib.cbook import boxplot_stats
from matplotlib.patches import Patch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

## Redirecting directory for loaded files

os.chdir(r"C:\Users\hl\Documents\python_data")
os.getcwd()

## Loading datasets into Python

acc = pd.read_csv('accounts.csv')
prod = pd.read_csv('products.csv')
sp = pd.read_csv('sales_pipeline.csv')
st = pd.read_csv('sales_teams.csv')

## Checking info and sample rows for datasets 

print("Accounts Dataset:")
print(acc.info())
print(acc.head())

print("\nProducts Dataset:")
print(prod.info())
print(prod.head())

print("\nSales Pipeline Dataset:")
print(sp.info())
print(sp.head())

print("\nSales Teams Dataset:")
print(st.info())
print(st.head())


# Data preprocessing

## Dropping duplicate rows from the dataset

acc = acc.drop_duplicates()
prod = prod.drop_duplicates()
sp = sp.drop_duplicates()
st = st.drop_duplicates()

## Creating 'company_size' column in accounts dataset

bins_0 = [0, 249, 999, 99999]
labels_0 = ["Small", "Medium", "Large"] 

acc["company_size"] = pd.cut(acc["employees"], bins=bins_0, labels=labels_0)
acc["company_size"] = acc["company_size"].cat.add_categories("Unknown")
acc["company_size"] = acc["company_size"].cat.add_categories("Others")

print("Accounts Pipeline Dataset:")
print(acc.head(10))


## Merge sales manager from sales team table to sales pipeline table

sp_draft1 = pd.merge(sp, st, on="sales_agent", how="left")

sp_draft2 = pd.merge(sp_draft1, prod, on="product", how="left")

final_sp = pd.merge(sp_draft2, acc, on="account", how="left")


## Checking and replacing null values

### Initial check for null values

print("\nNull Sums for Sales Pipeline Dataset:")
print(final_sp.isnull().sum())

### Check individual values of 'product' column in both final_sp table and product table

print("\nIndividual Values for Account Dataset:")
print(prod["product"].unique())
print("\nIndividual Values for Final Sales Pipeline Dataset:")
print(final_sp["product"].unique())

### Replacing the misspaced value in 'final_sp' table

sp["product"] = sp["product"].replace("GTXPro", "GTX Pro")

### Redoing the merge for the final_sp table

sp_draft1 = pd.merge(sp, st, on="sales_agent", how="left")

sp_draft2 = pd.merge(sp_draft1, prod, on="product", how="left")

final_sp = pd.merge(sp_draft2, acc, on="account", how="left")

### Filling in null values for all the remaining columns except engage_date, close_date and closing_period
final_sp = final_sp.fillna({
    "account": "Unknown",
    "company_size": "Unknown",
    "sector": "Unavailable",
    "year_established": "Unavailable",
    "revenue": 0,
    "employees": "Unavailable",
    "office_location": "Unavailable",
    "subsidiary_of": "Unavailable",
    "close_value": 0
})


### Final check for null sums

print("\nNull Sums for Sales Pipeline Dataset:")
print(final_sp.isnull().sum())

## Change engage date and close date to mixed format & adding close period column


final_sp["engage_date"] = pd.to_datetime(
    final_sp["engage_date"],
    format="mixed",
    dayfirst=True,
    errors="coerce"
)

final_sp["close_date"] = pd.to_datetime(
    final_sp["close_date"],
    format="mixed",
    dayfirst=True,
    errors="coerce"
)


closed_mask = final_sp["deal_stage"].isin(["Won", "Lost"])

final_sp["closing_period"] = np.where(
    closed_mask,
    (final_sp["close_date"] - final_sp["engage_date"]).dt.days,
    np.nan
)

valid_closed = final_sp[
    (closed_mask) &
    (final_sp["closing_period"] >= 0)
].copy()


# Sales History Data Analysis

## Company size against deal stage

### Grouping data
df = final_sp.groupby(["company_size", "deal_stage"]).size().unstack(fill_value=0)

### Plotting vertical bar chart
ax = df.plot(kind="bar")

### Add labels on bars
for container in ax.containers:
    ax.bar_label(container, label_type="edge")


### Add labels on axes and display plot

pt.xlabel("Company Size")
plt.ylabel("Count of Deals")
plt.title("Deal Stage by Company Size")
plt.legend(title="Deal Stage")
plt.xticks(rotation=0)
plt.show()


## Cilent sector against deal stage

### Filtering out rows with unavailable sector
filtered_sp = final_sp[final_sp["sector"] != "Unavailable"]

### Grouping data & sort sectors by total deals
df = filtered_sp.groupby(["sector", "deal_stage"]).size().unstack(fill_value=0)
df["Total"] = df.sum(axis=1)
df = df.sort_values("Total")
df = df.drop(columns="Total")

### Plotting horizontal bar chart
fig, ax = plt.subplots(figsize=(12,7))

df.plot(kind="barh", stacked=True, ax=ax)

### Add labels on bars
for container in ax.containers:
    ax.bar_label(container, label_type="center", fontsize=8)

### Add labels on axes and display plot
ax.set_xlabel("Count of Deals")
ax.set_ylabel("Company Sector")
ax.set_title("Deal Stage by Company Sector")

ax.xaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

ax.legend(title="Deal Stage", bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
plt.show()

## Cilent office location (country) against deal stage

### Filter only Won deals
won_df = final_sp[final_sp["deal_stage"] == "Won"]

### Count Won deals by country
country_counts = won_df["office_location"].value_counts()

### Get top 5 countries
top5 = country_counts.head(5)

### Sum the rest as "Others"
others = country_counts.iloc[5:].sum()

### Sort top 5 in descending order
top5_sorted = top5.sort_values(ascending=False)

### Combine top 5 + Others (Others always last)
final_counts = pd.concat([top5_sorted, pd.Series({"Others": others})])

### Plotting horizontal bar chart
fig, ax = plt.subplots(figsize=(10,6))
bars = ax.barh(final_counts.index, final_counts.values)

ax.invert_yaxis()  # highest on top
ax.set_title("Won Deals by Top 5 Countries (Others grouped)")
ax.set_xlabel("Count of Won Deals")
ax.set_ylabel("Country")

### Add value labels
ax.bar_label(bars, padding=3)

plt.tight_layout()
plt.show()

## Accounts against deal stage

### Filter dataset to Won and Lost deals
df = final_sp[
    (final_sp["account"] != "Unknown") &
    (final_sp["deal_stage"].isin(["Won", "Lost"]))
].copy()

### Find Top 10 accounts by total deals
top10_accounts = df["account"].value_counts().head(10).index

### Filter only Top 10 accounts
df_top10 = df[df["account"].isin(top10_accounts)]

### Count deals by account and deal stage
grouped = (
    df_top10
    .groupby(["account", "deal_stage"])
    .size()
    .unstack(fill_value=0)
)

### Sort by total deals
grouped["Total"] = grouped.sum(axis=1)
grouped = grouped.sort_values("Total")
grouped = grouped.drop(columns="Total")

### Plot stacked horizontal bar chart
fig, ax = plt.subplots(figsize=(12,8))

grouped.plot(
    kind="barh",
    stacked=True,
    ax=ax,
    color=["seagreen", "salmon"]
)

### Add labels
for container in ax.containers:
    ax.bar_label(container, label_type="center", fontsize=8)

## Setting titles and labels
ax.set_title("Won vs Lost Deals for Top 10 Accounts")
ax.set_xlabel("Number of Deals")
ax.set_ylabel("Account")

### Settings gridlines
ax.xaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

### Setting legend & display plot
ax.legend(title="Deal Stage", bbox_to_anchor=(1.02,1), loc="upper left")

plt.tight_layout()
plt.show()

## Closing period against deal stage

### Plotting boxplot chart
plt.figure(figsize=(8,6))

ax = sns.boxplot(
    data=valid_closed,
    x="deal_stage",
    y="closing_period",
    palette={"Won": "green", "Lost": "red"}
)

### Adding title and axes labels
plt.title("Closing Period Distribution by Deal Stage")
plt.xlabel("Deal Stage")
plt.ylabel("Closing Period (days)")
plt.grid(axis="y", alpha=0.3)

### Define deal stages
stages = ["Won", "Lost"]

### Looping stages & extracting data 
for i, stage in enumerate(stages):

    values = valid_closed.loc[
        valid_closed["deal_stage"] == stage,
        "closing_period"
    ].dropna()

    if len(values) == 0:
        continue

    stats = boxplot_stats(values)[0]

    lower_whisker = stats["whislo"]
    median = stats["med"]
    upper_whisker = stats["whishi"]

   # Median label
    ax.text(i, median, f"Median: {median:.0f}",
            ha="center", va="bottom", fontsize=9, color="blue")

    # Upper whisker label (NOT max)
    ax.text(i, upper_whisker, f"Upper whisker: {upper_whisker:.0f}",
            ha="center", va="bottom", fontsize=8, color="red")

    # Lower whisker label
    ax.text(i, lower_whisker, f"Lower whisker: {lower_whisker:.0f}",
            ha="center", va="top", fontsize=8, color="green")

plt.show()

## Count of won and lost deals against closing period

### Filter only Won deals
won_df = final_sp[final_sp["deal_stage"] == "Won"].copy()
won_df = won_df.dropna(subset=["closing_period"])
won_df = won_df[won_df["closing_period"] >= 0]

### Create bins of width = 25 days
bins = np.arange(
    0,
    won_df["closing_period"].max() + 25,
    25
)

### Plotting histogram for count of won deals against closing period
fig, ax = plt.subplots(figsize=(10,6))

counts, bin_edges, patches = ax.hist(
    won_df["closing_period"],
    bins=bins,
    edgecolor="black"   # black edges
)

### Adding labels to bar chart
for count, patch in zip(counts, patches):
    if count > 0:
        ax.text(
            patch.get_x() + patch.get_width()/2,
            patch.get_height(),
            int(count),
            ha="center",
            va="bottom",
            fontsize=10
        )

ax.set_title("Distribution of Closing Period for Won Deals (25-day bins)")
ax.set_xlabel("Closing Period (Days)")
ax.set_ylabel("Number of Won Deals")

plt.show()

### Filter only Lost deals
lost_df = final_sp[final_sp["deal_stage"] == "Lost"].copy()
lost_df = lost_df.dropna(subset=["closing_period"])
lost_df = lost_df[lost_df["closing_period"] >= 0]

### Create bins of width = 25 days
bins = np.arange(
    0,
    lost_df["closing_period"].max() + 25,
    25
)

### Plotting histogram for count of lost deals against closing period
fig, ax = plt.subplots(figsize=(10,6))

counts, bin_edges, patches = ax.hist(
    lost_df["closing_period"],
    bins=bins,
    edgecolor="black"   # black edges
)

### Adding labels to bar chart
for count, patch in zip(counts, patches):
    if count > 0:
        ax.text(
            patch.get_x() + patch.get_width()/2,
            patch.get_height(),
            int(count),
            ha="center",
            va="bottom",
            fontsize=10
        )

ax.set_title("Distribution of Closing Period for Lost Deals (25-day bins)")
ax.set_xlabel("Closing Period (Days)")
ax.set_ylabel("Number of Lost Deals")

plt.show()

## Count of won and lost deals against product

### Filter only closed deals (Won or Lost)
df = final_sp[
    final_sp["deal_stage"].isin(["Won", "Lost"])
].copy()

### Count deals by product and deal_stage
grouped = (
    df.groupby(["product", "deal_stage"])
    .size()
    .unstack(fill_value=0)
)

### Sort by total closed deals
grouped["Total"] = grouped.sum(axis=1)
grouped = grouped.sort_values("Total")
grouped = grouped.drop(columns="Total")

### Plot stacked horizontal bar chart
fig, ax = plt.subplots(figsize=(12,8))

grouped.plot(
    kind="barh",
    stacked=True,
    ax=ax,
    color=["seagreen", "salmon"]  # Won = green, Lost = red
)

### Adding value labels
for container in ax.containers:
    ax.bar_label(container, label_type="center", fontsize=8)

### Titles and labels
ax.set_title("Closed Deals by Product (Won vs Lost)")
ax.set_xlabel("Number of Deals")
ax.set_ylabel("Product")

### Setting gridlines
ax.xaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

### Legend on right
ax.legend(title="Deal Stage", bbox_to_anchor=(1.02,1), loc="upper left")

plt.tight_layout()
plt.show()

## Product against closing value for won deals

### Filter to only 'Won' deals
df_won = final_sp[final_sp["deal_stage"] == "Won"]

### Split deals into value tiers
low_df = df_won[df_won["close_value"] < 2000]
mid_df = df_won[(df_won["close_value"] >= 2000) & (df_won["close_value"] <= 10000)]
high_df = df_won[df_won["close_value"] > 10000]


### Plotting boxplot chart
def plot_box_with_median_clean(data, ax, title):
    sns.boxplot(data=data, x="product", y="close_value", ax=ax)
    ax.set_title(title)

### Grouping data & adding labels
    grouped = data.groupby("product")["close_value"]

    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * 0.03   # spacing for labels

    for i, (product, values) in enumerate(grouped):
        med = values.median()
        q3 = values.quantile(0.75)
        y_pos = q3 + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02

        ax.text(
            i,
            q3 + offset,           
            f"Med={med:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
        )

    ax.tick_params(axis="x", rotation=30)

### Plotting each tier
fig, axes = plt.subplots(1, 3, figsize=(18,6), sharey=False)

plot_box_with_median_clean(low_df, axes[0], "Low Value Products (<2000) – Won Deals")
plot_box_with_median_clean(mid_df, axes[1], "Medium Value Products (2000–10000) – Won Deals")
plot_box_with_median_clean(high_df, axes[2], "High Value Products (>10000) – Won Deals")

plt.tight_layout()
plt.show()

## Boxplot and scatterplot for company size against closing value for won deals

### Filtering data

df = final_sp[
    (final_sp["deal_stage"] == "Won") &
    (final_sp["company_size"].notna()) &
    (final_sp["close_value"].notna()) &
    (~final_sp["company_size"].isin(["Unknown", "Others"]))
].copy()

### Fix categorical order

#### Convert to string to remove hidden categories
df["company_size"] = df["company_size"].astype(str)

#### Define logical order
order = ["Small", "Medium", "Large"]

#### Keep only valid sizes (extra safety)
df = df[df["company_size"].isin(order)]

### Plotting boxplot

plt.figure(figsize=(8,6))

ax = sns.boxplot(
    data=df,
    x="company_size",
    y="close_value",
    order=order,
    showfliers=True
)

### Compute true boxplot stats per group

groups = [
    df[df["company_size"] == size]["close_value"].values
    for size in order
]

stats = cbook.boxplot_stats(groups)

### Adding labels

for i, stat in enumerate(stats):

    median = stat["med"]
    whisker_low = stat["whislo"]
    whisker_high = stat["whishi"]

    # Median label
    ax.text(i, median, f"Median: {median:.0f}",
            ha="center", va="bottom",
            fontsize=9, color="blue")

    # Upper whisker label
    ax.text(i, whisker_high, f"Upper whisker: {whisker_high:.0f}",
            ha="center", va="bottom",
            fontsize=8, color="red")

    # Lower whisker label
    ax.text(i, whisker_low, f"Lower whisker: {whisker_low:.0f}",
            ha="center", va="top",
            fontsize=8, color="green")

### Add titles and formatting

ax.set_title("Closing Value by Company Size (Won Deals)")
ax.set_xlabel("Company Size")
ax.set_ylabel("Closing Value")

plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

## Cilent company sector against closing value for won deals

### Filtering data
df = final_sp[
    (final_sp["deal_stage"] == "Won") &
    (final_sp["sector"].notna()) &
    (final_sp["close_value"].notna())
].copy()

### Plotting boxplot
plt.figure(figsize=(8,6))
ax = sns.boxplot(
    data=df,
    x="sector",
    y="close_value",
    showfliers=True
)

### Compute true boxplot stats per group
groups = [g["close_value"].values for _, g in df.groupby("sector")]
stats = cbook.boxplot_stats(groups)

for i, stat in enumerate(stats):
    median = stat["med"]
    whisker_low = stat["whislo"]
    whisker_high = stat["whishi"]

    # Median label
    ax.text(i, median, f"Median: {median:.0f}",
            ha="center", va="bottom", fontsize=9, color="blue")

    # Upper whisker label (NOT max)
    ax.text(i, whisker_high, f"Upper whisker: {whisker_high:.0f}",
            ha="center", va="bottom", fontsize=8, color="red")

    # Lower whisker label
    ax.text(i, whisker_low, f"Lower whisker: {whisker_low:.0f}",
            ha="center", va="top", fontsize=8, color="green")

ax.set_title("Closing Value by Cilent Sector (Won Deals)")
ax.set_xlabel("Sector")
ax.set_ylabel("Closing Value")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


## Closing period against closing value

### Ensure closing_period is non-negative and non-zero
valid_closed = final_sp[
    (closed_mask) &
    (final_sp["closing_period"] >= 0)
].copy()

### Filtering data and grouping by subset
df_won = valid_closed[
    valid_closed["deal_stage"] == "Won"
].dropna(subset=["closing_period", "close_value", "company_size"])

### Plotting scatterplot graph
plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df_won,
    x="closing_period",
    y="close_value",
    hue="company_size",
    palette="viridis",
    alpha=0.6
)

plt.title("Closing Period vs Close Value (Won Deals)")
plt.xlabel("Closing Period (days)")
plt.ylabel("Close Value")
plt.legend(title="Company Size")
plt.show()

## Top 10 accounts against average closing value

### Filter Won deals
df_won = final_sp[final_sp["deal_stage"] == "Won"].dropna(
    subset=["account", "close_value", "company_size"]
)

### Average close value per account
avg_close = df_won.groupby("account")["close_value"].mean()

### Get company size per account
account_size = df_won.groupby("account")["company_size"].agg(lambda x: x.mode()[0])

### Combine both
account_df = pd.DataFrame({
    "avg_close": avg_close,
    "company_size": account_size
})

### Top 10 accounts
top10 = account_df.sort_values("avg_close", ascending=False).head(10)

### Remaining accounts grouped into Others
others_value = account_df.iloc[10:]["avg_close"].mean()

others_row = pd.DataFrame(
    {"avg_close":[others_value], "company_size":["Others"]},
    index=["Others"]
)

plot_df = pd.concat([top10, others_row])

### Color mapping
color_map = {
    "Small": "skyblue",
    "Medium": "orange",
    "Large": "green",
    "Others": "gray"
}

### Add color column BEFORE sorting
plot_df["color"] = plot_df["company_size"].map(color_map)

### Sort for horizontal bar chart
plot_df = plot_df.sort_values("avg_close")

### Plot horizontal bar chart
fig, ax = plt.subplots(figsize=(9,6))

ax.barh(
    plot_df.index,
    plot_df["avg_close"],
    color=plot_df["color"]
)

ax.set_xlabel("Average Close Value")
ax.set_ylabel("Account")
ax.set_title("Average Close Value by Top 10 Accounts")

### Value labels
for i, v in enumerate(plot_df["avg_close"]):
    ax.text(v, i, f"{v:,.0f}", va="center", ha="left")

### Legend on right side
legend_elements = [
    Patch(facecolor=color_map[size], label=size)
    for size in color_map
]

ax.legend(
    handles=legend_elements,
    title="Company Size",
    loc="center left",
    bbox_to_anchor=(1, 0.5)
)

plt.tight_layout(rect=[0,0,0.85,1])
plt.show()

### Add labels
for container in ax.containers:
    ax.bar_label(container, fmt="%.0f", padding=3)

from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor=color_map[size], label=size)
    for size in color_map
]

ax.legend(handles=legend_elements,
          title="Company Size",
          loc="center left",
          bbox_to_anchor=(1,0.5))

plt.tight_layout()
plt.show()


## Revenue of client company against closing value for won deals

### Filtering data and grouping by subset
df_won = final_sp[
    (final_sp["deal_stage"] == "Won")
].dropna(subset=["revenue", "close_value", "company_size"])


### Plotting scatterplot graph
plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df_won,
    x="revenue",
    y="close_value",
    hue="company_size",
    palette="viridis",
    alpha=0.7
)

plt.xlabel("Company Revenue")
plt.ylabel("Close Value")
plt.title("Revenue vs Close Value by Company Size (Won Deals)")
plt.legend(title="Company Size")
plt.show()

## Linear regression for revenue of client company against closing value

### Filter to Won deals only
df_won = final_sp[final_sp["deal_stage"] == "Won"].copy()

### Keep only needed columns and drop NaNs
df_won = df_won[["revenue", "close_value"]].dropna()

## Input data for linear regression model
X = df_won["revenue"].values.reshape(-1, 1)
y = df_won["close_value"].values

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

### Print R^2 score
print("R^2 score:", r2)

## Plotting scatterplot chart
plt.figure(figsize=(8,6))
plt.scatter(X, y, alpha=0.5, label="Actual data")

# Regression line (sort X for clean line)
sorted_idx = np.argsort(X.flatten())
plt.plot(X[sorted_idx], y_pred[sorted_idx], color="red", linewidth=2, label="Regression line")

plt.xlabel("")
plt.ylabel("Close Value")
plt.title(f"Revenue of Cilent Company vs Close Value (Won Deals)\nR² = {r2:.3f}")

plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# Sales Team Performance Analysis

## Deal stage against region of sales team

### Grouping data and creating helper column
df = final_sp.groupby(["regional_office", "deal_stage"]).size().unstack(fill_value=0)
df["Total"] = df.sum(axis=1)
df = df.sort_values("Total")
df = df.drop(columns="Total")

### Plotting horizontal stacked bar chart
fig, ax = plt.subplots(figsize=(12,7))

df.plot(kind="barh", stacked=True, ax=ax)

for container in ax.containers:
    ax.bar_label(container, label_type="center", fontsize=8)

ax.set_xlabel("Count of Deals")
ax.set_ylabel("Sales Office Region")
ax.set_title("Deal Stage by Sales Office Region")

ax.xaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

ax.legend(title="Deal Stage", bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
plt.show()

## Deal stage against manager of sales team

### Define order of deal stages
deal_stages = ["Engaging", "Prospecting", "Lost", "Won"]

### Looping each regional office
for region, df_region in final_sp.groupby("regional_office"):
    
    ### Group by manager and deal stage
    grouped = df_region.groupby(["manager", "deal_stage"]).size().unstack(fill_value=0)

    ### Ensure all deal stages exist
    for stage in deal_stages:
        if stage not in grouped.columns:
            grouped[stage] = 0

    ### Reorder columns
    grouped = grouped[deal_stages]

    ### Sort managers by total deals
    grouped["Total"] = grouped.sum(axis=1)
    grouped = grouped.sort_values("Total", ascending=False)
    grouped = grouped.drop(columns="Total")

    ### Plotting bar charts
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(grouped.index))
    width = 0.2

    for i, stage in enumerate(deal_stages):
        bars = ax.bar(x + i*width, grouped[stage], width, label=stage)
        ax.bar_label(bars, padding=2, fontsize=8)

    ### Adding labels and titles
    ax.set_title(f"Deal Stage by Sales Manager (Region: {region})")
    ax.set_xlabel("Sales Manager")
    ax.set_ylabel("Count of Deals")

    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(grouped.index, rotation=45, ha="right")

    ax.legend(title="Deal Stage")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    ### Display plot
    plt.tight_layout()
    plt.show()

## Average sales value against manager

### Filtering and averaging data
won_df = final_sp[final_sp["deal_stage"] == "Won"].copy()
manager_avg = (
    won_df
    .groupby(["manager", "regional_office"])["close_value"]
    .mean()
    .reset_index(name="avg_deal_value")
    .sort_values("avg_deal_value", ascending=False)
)

print(manager_avg.head())

### Plotting horizontal bar chart
fig, ax = plt.subplots(figsize=(10,6))

sns.barplot(
    data=manager_avg,
    y="manager",
    x="avg_deal_value",
    hue="regional_office",
    dodge=False,
    ax=ax
)

### Adding labels & titles
for container in ax.containers:
    ax.bar_label(container, fmt="%.0f", padding=3)

ax.set_title("Average Deal Value by Sales Manager (Colored by Region)")
ax.set_xlabel("Average Deal Value")
ax.set_ylabel("Sales Manager")
ax.legend(title="Region", bbox_to_anchor=(1.02,1), loc="upper left")
ax.grid(axis="x", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

## Total sales value of sales manager

### Filter to Won deals
won_df = final_sp[final_sp["deal_stage"] == "Won"].copy()

### Group by manager and region, sum close_value
manager_region_sales = (
    won_df
    .groupby(["manager", "regional_office"])["close_value"]
    .sum()
    .reset_index(name="total_sales_value")
)

### Sort by total sales value
manager_region_sales = manager_region_sales.sort_values("total_sales_value", ascending=False)

print(manager_region_sales.head())

### Plotting bar chart
fig, ax = plt.subplots(figsize=(10,6))

sns.barplot(
    data=manager_region_sales,
    y="manager",
    x="total_sales_value",
    hue="regional_office",
    dodge=False,
    ax=ax
)

for container in ax.containers:
    ax.bar_label(container, fmt="%.0f", padding=3)

ax.set_title("Total Sales Value by Sales Manager (Colored by Region)")
ax.set_xlabel("Total Sales Value")
ax.set_ylabel("Sales Manager")
ax.legend(title="Region", bbox_to_anchor=(1.02,1), loc="upper left")
ax.grid(axis="x", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()


## Average sales value of agents against sales manager

### Filter Won deals
won_df = final_sp[final_sp["deal_stage"] == "Won"].copy()

### Count deals per agent
deal_counts = (
    won_df
    .groupby(["sales_agent", "manager"])["opportunity_id"]
    .count()
    .reset_index(name="deal_count")
)

### Average deal size per agent
avg_values = (
    won_df
    .groupby(["sales_agent", "manager"])["close_value"]
    .mean()
    .reset_index(name="avg_deal_size")
)

### Merge them together
agent_stats = pd.merge(deal_counts, avg_values,
                       on=["sales_agent", "manager"],
                       how="inner")

print(agent_stats.head())

### Plotting scatterplot chart
plt.figure(figsize=(10,7))

sns.scatterplot(
    data=agent_stats,
    x="deal_count",
    y="avg_deal_size",
    hue="manager",
    palette="tab10",
    s=120,
    alpha=0.8
)

plt.xlabel("Number of Won Deals")
plt.ylabel("Average Deal Size")
plt.title("Deal Count vs Average Deal Size (Colored by Manager)")

plt.legend(title="Manager", bbox_to_anchor=(1.02,1), loc="upper left")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

## Average closing period against sales manager

### Filter Won deals
df_won = final_sp[
    (final_sp["deal_stage"] == "Won") &
    (final_sp["closing_period"].notna()) &
    (final_sp["manager"].notna())
]

### Compute average closing period and keep region info
avg_closing = (
    df_won.groupby(["manager", "regional_office"])["closing_period"]
    .mean()
    .reset_index()
)

### Sort by closing period
avg_closing = avg_closing.sort_values("closing_period")

### Create color map for regions
regions = avg_closing["regional_office"].unique()
colors = plt.cm.Set2.colors[:len(regions)]
color_map = dict(zip(regions, colors))

bar_colors = avg_closing["regional_office"].map(color_map)

### Plotting horizontal bar chart
fig, ax = plt.subplots(figsize=(8,5))

bars = ax.barh(avg_closing["manager"], avg_closing["closing_period"], color=bar_colors)

ax.set_xlabel("Average Closing Period (days)")
ax.set_ylabel("Manager")
ax.set_title("Average Sales Velocity by Manager")

### Add value labels
ax.bar_label(bars, fmt="%.1f", padding=3)

### Create legend
handles = [
    plt.Line2D([0], [0], color=color_map[r], lw=8) for r in regions
]

ax.legend(handles, regions, title="Region", bbox_to_anchor=(1.02,1), loc="upper left")

plt.tight_layout()
plt.show()