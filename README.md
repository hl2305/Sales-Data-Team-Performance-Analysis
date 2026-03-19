# Sales Data & Team Performance Analysis (Python)

## Data Overview
The objective of this project is to investigate the sales history data and sales team performance data of a technology company in order to identify patterns and create improvements and recommendations that aims to optimize the sales pipeline efficiency and identifying key factors of sales performance.

The data is sourced from [Kaggle](https://www.kaggle.com/datasets/innocentmfa/crm-sales-opportunities/data) as a sample dataset for skill practice for data analysts. The data was split into 5 tables: accounts, products, sales_pipeline, sales_teams and data_dictionary. Each table are relational to each other, with the entity relationship diagram shown as follows:

<img width="500" height="800" alt="ER diagram" src="https://github.com/user-attachments/assets/c2a8a315-f4c2-4041-9d17-32429b88464c" />

This project is viewable in two different format, one is pairing the GitHub ReadMe with the .py file for code, as well as the Jupyter Notebook file as attached in the repository.

## Prepare Phase
### Importing libraries
The libraries used in the analysis include:
- pandas for managing data
- numpy for mathematical operations
- sklearn for machine learning and machine-learning related functions
- seaborn for data visulizations
- matplotlib for additional plotting functions

### Filling null values

The four tables were merged together to simplify the analysis process into the `final_sp` table. The table was checked for null values and it shows that 14 of these columns contain null values. 
- For the `account` column, some deals are yet to be closed, such as deals that are in the Engaging or Prospecting stage, where the client is not identified yet. This also leads to columns such as `company_size`, `sector`, `year_established`, `revenue`, `employees`, `office_location` and `subsidiary_of` to be null.

- As engaging and prospecting deals are not closed yet, this also leads to the columns `close_date`, `closing_period` and `close_value` to be null. As for prospecting deals only, the `engage_date` column is yet to be identified.

- It is odd that the `series` and `sales_price` column would have null values, as every column of the `product` column in the final_sp table are filled. The data under the `product` column was checked to have a misspacing error, leading to errors while merging the tables. Thus, this error was fixed by replacing the misspaced value with the correct value and remerging the tables.

- All the null values in these columns were filled except the `engage_date` and `close_date` columns to avoid data errors.

### Adding columns

Two columns were added into the merged 'final_sp' table to derive better insights:

- `company_size` column was added to classify the accounts into small (1-249 employees), medium (250-999 employees) and large companies (1000+ employees). 

- `closing_period` was added to determine the number of days between the engage date and close date for deals (only applicable to Won and Lost deals).

## Sales Data Analaysis
To identify trends such as best-selling products, factors affecting win-loss rate in closing deals and top customers, exploratory data analysis was performed on the dataset.
With these insights, improvements and recommendations can be suggested in order to improve future sales and provide better customer service.

### a) Investigate patterns and trends from the dataset related to conversion rate of deals.
#### i) Identify trends of company size against number of deals and win/loss rate of closed deals.
<img width="700" height="1050" alt="1" src="https://github.com/user-attachments/assets/1134cef8-defa-49b9-bd40-c9af835caf84" />

As there are a total of 8,800 deals in the dataset, around **69.17%** of the deals (6,087 deals) involves larger companies, which sizes more than 1,000 employees based on the `company_size` category. Other than that, **6.66%** of the total deals involves small companies (1-250 employees) whereas **7.98%** of the total deals involves medium companies (251-1000 employees). As for the other 16.2%, the `company_size` is labelled as 'Unknown' and the deals are in Engaging/Prospecting deal stage, where the client company is yet to be identified.

In terms of closed deals which only considers won and lost deals, there is a **62.9% win rate** for both large and medium companies. As for smaller companies, the win rate for closed deals is around **65.7%** although with lesser number of deals.   

#### ii) Which client sectors bring in the most number of deals, and how is the win/loss rate of the closed deals based on client sector?
<img width="700" height="1050" alt="2" src="https://github.com/user-attachments/assets/7ebd8320-2e35-4f8d-84e0-5b6b97aab413" />

**15.8%** of the total deals come from the retail sector, with **13.24%** of deals from the technology sector trailing at second place and **11.94%** of deals are from the medical sector in third place. The cilent company has not been stated in around 16.2% of the deals, thus labelled as 'Unavailable'. 

As for the closed deals, the win/loss rate of all the sectors are similar to each other, ranging to **61%-65% win/loss rate**. Therefore, it may be more strategical to focus more on the high-demand sectors (retail, technology, medical, software and finance) while also generating new leads.

#### iii) For won deals, which country are most of the cilents from?
<img width="700" height="1050" alt="3" src="https://github.com/user-attachments/assets/e86fc82d-56e4-40fe-bb5b-e54094b388d9" />

It can be seen that **82.9% won deals** come from the **United States**, **7.1% won deals** are from the **other top 5 countries (Korea, Panama, Belgium and Italy)** while **9.9%** come from other countries combined. It is important to maintaining the customer base in United States, while also looking for potential new cilents outside of United States.

#### iv) Which accounts have the most closed deals (both won and lost)?
<img width="700" height="1050" alt="4 5" src="https://github.com/user-attachments/assets/6f458e7e-c7c3-43f1-97c3-93ed81454ddb" />

The top 5 accounts that have the most number of closed deals are **Hottechi** (193 deals), **Kan-code** (187 deals), **Konex** (171 deals), **Condax** (159 deals) and **Dontechi** (117 deals). The accounts that have lesser deals than Singletechno (top 10 account) are not included in the horizontal bar chart. 
Among the top 10 accounts, Singletechno has the **highest win rate** of **66%** among the top 10 accounts, whereas Hottechi has the **lowest win rate** of **57.5%** among the top 10 accounts. 

#### v) Is there a specific amount of closing period (period from engage date to closing date) that may lead to higher win rate?
<img width="700" height="1050" alt="6 1" src="https://github.com/user-attachments/assets/c057db5d-f55b-4ecc-b3be-40b571052ad4" />
<img width="700" height="1050" alt="6 2" src="https://github.com/user-attachments/assets/3702c2bf-3fd6-42eb-84e6-0816a0ee3349" />

The histogram for closing period against number of closed deals shows a **similar trend** for both won and lost deals, where **most** of the deals close **within a 25-day period**, **followed by** the range of 75-100 days. Won deals still see significantly more success for a closing period of **within 125 days** before seeing a **sharp decline** in won deals, whereas this sharp decline is seen in lost deals after 100 days. 

This indicates that closing period is not a strong indicator of achieving higher success rate. Therefore, it is important for the sales agents and sales manager to evaluate on the deal process such as negotiation processes and also set priorities based on other factors like potential deal value to maximize efficiency in the sales pipeline.

### b) Which products are the most popular among cilents? How is the win/loss rate based on products?
<img width="700" height="1050" alt="7" src="https://github.com/user-attachments/assets/e40faecc-0df0-4753-adcd-a048dd344caa" />

It is seen that **GTX Basic** is the most popular product among cilents, with a total closed deal count of 1436 deals, followed by **MG Special** with 1223 deals and **GTX Pro** with 1147 deals. 	

For win/loss ratio for the products, the product with the highest **win/loss ratio** is the **GTX Plus Pro** with a win rate of **64.3%**, although it ranks the second lowest in terms of total closed deal count. As for the **lowest win rate** product, it is the **GTK 500** with a **60% win rate** and only total closed deal count of 25 deals understandably with its higher price point, with the MG Advanced coming in close with a 60.3% win rate with its higher closed deal count of 1084 deals. 

It is important to perform regular market surveys to keep updated on market prices so that the company can have a competitive edge in the market, especially for products with lower success rate. Another way is to conduct market research to offer new innovative products with reasonable prices to potentially grab market share. 

### c) Is there any factors that affect the close value for each won deal?
#### i) Investigate the close value for won deals based on product type and series.
<img width="700" height="1050" alt="8" src="https://github.com/user-attachments/assets/9976c639-7dea-4704-8e28-e506b291fe8b" />

The data was separated into 3 separate boxplots for this problem statement, as the price of the products that varies significantly can affect the viewability if all the products were to be included in one same boxplot. The data was separate into low-value products (value below `$2000`), medium-value products (value above `$2000` and below `$10000`) and high-value products (value above `$10000`). 

For all the products would be given during negotiations with the cilents, with the highest price deviation of the close value compared to the sales price is the GTK 500 with **around 3.3% below the sales price** for its median closing value. It is also noted that the lowest priced MG Special has a closing value median of `$55` which is equal to its price points including low-value, medium-value and high-value products, the median point for the close value **averages around 1 product per deal**. Some price deviations are expected as slight discount will be given after negotiations.

#### ii) Does larger companies usually have larger deal values for closed deals?
<img width="700" height="1050" alt="9" src="https://github.com/user-attachments/assets/a292473f-c5c3-43aa-9321-17739a168f53" />

From the boxplot, it was found out that **small companies tend to have smaller deal values** when compared to medium and large companies. However, the average deal values for medium and large companies are quite similar to each other. Other than that, it must also be noted that larger companies tend to buy more products in bulk or purchase more expensive products as they have higher deal values as seen with the anomaly dots above the box large companies where the deals are valued `$20,000` and above. 

#### iii) What are the sectors that have larger deal values?
<img width="700" height="1050" alt="10" src="https://github.com/user-attachments/assets/d9c0a112-c1ff-4009-805a-94befb6dd943" />

It is observed that the **median point** of the close value for each sector are quite similar to each other, ranging from `$1079` to `$1171`. However, the **upper whisker** for the **retail and marketing sectors** are **slightly more** than other sectors at around `$7,300`, whereas the upper whisker for the **software sector** is the **lowest** at `$6,284`. This may indicate that software has lesser demand for the products, or the software sector has lower requirements for specifications of the products, opting for cheaper products. Last but not least, it can be seen that the **entertainment sector** has the **most amount of deals valued** over `$20,000` with 4 deals, whereas the **technology and services sector** have **no deals above** `$20,000`. Thus, it is important to prioritize the deals that has more potential deal value without compromising too much lead time.

#### iv) Does closing period have any correlation with the closing value?
<img width="700" height="1050" alt="11" src="https://github.com/user-attachments/assets/4c1ab460-cad7-4c57-b701-5a6a87eca087" />

From the scatterplot, it can be seen that there is **not much correlation between closing period and closing value**. The majority of close deals ranging up to around `$7000-$8000` in close value take around within 150 days to complete and in rare cases, can even take up to `$6000` to close. On the other hand, it is seen that deals that are large in closing value (`$20,000` and above) can be completed even within 20 days, especially within large companies that have better cashflow. There are also 2 medium companies that closed deals valued above `$20,000` but it took around 200-300 days to close the deal.

#### v) Which accounts have higher average closing value?
<img width="700" height="1050" alt="12" src="https://github.com/user-attachments/assets/a23b13b0-b1f3-4364-a6f2-94f1e35d08e3" />

It can be observed that **8 out of the top 10 accounts** that have a **higher average deal value** are large companies, while the other 2 are medium companies. It should be noted that there is **no significant difference** of the average close value between the top 10 accounts, where Xx-holding has the highest average deal value of `$3,528` and Groovestreet has the least average deal value of `$3,031` among the top 10 accounts. 

#### vi) Do accounts with higher revenue have higher value of closed deals?
<img width="700" height="1050" alt="13 1" src="https://github.com/user-attachments/assets/57fe748c-c320-4fe7-aa33-22403383310c" />

It is seen that the pattern of the scatterplot for the revenue of accounts against closing value are **quite similar** to that of **the graph of company size against closing value**. It is observed that small companies have revenue of less than `$100k`, medium companies have revenue less than `$500k`, whereas large companies have revenues ranging from `$400k` up to `$11m`. This does not affect the closing value in any way, except that medium companies and large companies sometimes purchase big orders as seen with the closing value above `$20000`. 

<img width="700" height="1050" alt="13 2" src="https://github.com/user-attachments/assets/711e3f97-4978-4666-ae0b-fe7e69c79f3e" />

A linear regression model was attempted on the following scatterplot. The R^2 score was found to be a value of 0.002, which shows that it has little to no correlation between revenue of the account company and the deal value of won deals.

## Team Performance Data Analaysis
Team performance was also analysed in this dataset to identify top-performing as well as under-performing teams as efforts to improve the sales pipeline as well as encourage company growth.

### Identify and evaluate the performance of each sales team under different sales manager in the sales pipeline.
#### i) Which region perform the best in sales?
<img width="700" height="1050" alt="14" src="https://github.com/user-attachments/assets/9e831886-54e0-45aa-b109-cc980a3663fb" />

The **Central** region performs the best in sales, with 1629 won deals and 975 lost deals, resulting in a **63.7% win rate**. The **West** region has 1438 won deals and 811 lost deals which is a **63.9% win rate**, whereas the **East** region has the lowest amount of deals and have a **63.0% win rate** with 1171 won deals and 687 lost deals. It must also be pointed out although each region are actively engaging accounts for new deals, it is only the Central region is prospecting deals and performing initiatives to look for new customers.

#### ii) How is the conversion rate for each sales manager from each team?
<img width="700" height="1050" alt="15" src="https://github.com/user-attachments/assets/fb7c190e-7825-48f4-a46b-137d96c50383" />
<img width="700" height="1050" alt="16" src="https://github.com/user-attachments/assets/e30adbcb-be4d-4b89-beb4-48aab36bfe26" />
<img width="700" height="1050" alt="17" src="https://github.com/user-attachments/assets/2bf55c6c-68d1-4a3c-8545-255ac8762491" />

**Manager summary**:

- **Melvin Marxen (Central)**: 62.2% win rate, most won deals (882 deals), most prospecting deals overall (296 deals) and 11% more engaging deals than Dustin (215 deals)

- **Dustin Brinkmann (Central)**: 63% win rate, 15% lesser won deals than Melvin (747 deals)

- **Rocco Neubert (East)**: 62.1% win rate, 44% more deals than Cara (691 deals), around same amount of engaging deals with Cara

- **Cara Losch (East)**: 64.4% win rate, least won deals overall (480 deals)

- **Summer Sewald (West)**: 64.3% win rate, 36% more won deals than Celia (828 deals), most engaging deals overall (414 deals)

- **Celia Rouche (West)**: 63.4% win rate, 20% less engaging deals than Summer (334 deals)

**Overall insights**: 
- West region has the most engaging deals compared to Central region and East region which are similar to each other
- Only Central region has prospecting deals while West and East has none. This may indicate a strategical difference between the Central region and the other regions.
- The conversion rate across all managers are quite uniform to each other, with only a maximum difference of 2% between Melvin Marxen and Cara Losch.

#### iii) What is the average sales value and total sales value for each manager?
<img width="700" height="1050" alt="18" src="https://github.com/user-attachments/assets/e5041687-d3ef-451c-bad0-9f05b40463a1" />
<img width="700" height="1050" alt="19" src="https://github.com/user-attachments/assets/71d449e4-2a9b-47dc-b7c7-295b928cdbe0" />

- **Rocco Neubert (East)** has the **highest average deal value** among all the managers with an average of `$2837` per won deal whereas **Dustin Brinkmann (Central)** has the **least average deal value** of `$1465` although coming in third in terms of amount of won deals. This may indicate that the majority of Dustin Brinkmann's deals may be more focused on lower value products such as the MG Special. 

- **Melvin Marxen (Central)** has the **highest total deal value** of `$2,251,930` which is **double of that** compared to **Dustin Brinkmann (Central)** with the **lowest total deal value** of `$1,094,363`. The top performers of their own region, **Summer Sewald (West)** and **Rocco Neubert (East)** are **similar to each other** in total deal value, which is `$1,964,750` and `$1,960,545` respectively who came in second and third overall. **Cara Losch (East)** has a total deal value of `$1,130,049` which is **barely above Dustin Brinkmann (Central)** although having the lowest amount of deals won.

#### iv) How does each sales agent under the managers perform?
<img width="700" height="1050" alt="20" src="https://github.com/user-attachments/assets/7c85fadb-3997-495f-8218-862fe5bf525a" />

- Although **Melvin Marxen (Central)** has the highest total deal value and amount of deals won, it is observed that based on the sales agent performance scatterplot, there is **only 1 standout sales agent** who fetched around 350 won deals and average deal size of around `$3250`, whereas the **other sales agents** under Melvin Marxen **perform below average** with under 200 won deals and lower average deal size ranging around `$1600-$2600`. 

- **No agents under Dustin Brinkmann (Central)** has **achieved an average deal value of over** `$2200`, and **only one** of them has **completed more than 200 won deals**.

- Teams under **Summer Sewald (West)** and **Rocco Neubert (East)** seem to have a **balanced performance across all sales agents** in terms of won deals. This can suggest strong systematic business processes and teamwork.

- Teams under **Celia Rouche (West)** and **Cara Losch (East)** have a more **imbalanced trend** in terms of **average deal value**, suggesting that different agents may focus on different products when dealing with cilents.

#### v) What is the average sales velocity for won deals under each manager?
<img width="700" height="1050" alt="21" src="https://github.com/user-attachments/assets/0fe344dd-75e7-4ff5-983c-ef6f0ac863c3" />

**Melvin Marxen (Central)** is the **slowest performer** with an average closing period of 45.8 days whereas the **fastest performer** is **Rocco Neubert (East)** who closed successful deals within 37 days. The **difference** in the sales velocity with the period of **7-8 days** indicate that the sales pipeline processes are standardized and efficient.

## Recommendations

#### Prioritize High-Demand Sectors for Lead Generation
- Sectors such as Retail, Technology, Medical, Software and Finance proved to be high-demand sectors with a consistent win rate of 61-65%. Thus, the company could leverage this by allocating more marketing and sales resources to these sectors and creating occasional promotions to pull more customers in.

#### Expanding Geographic Diversification

- As 82% of the won deals are from United States, the company should scale up to international market development once the business growth and cashflow is sufficient. The company should build regional partnerships outside of the U.S, however manufacturing, logistics and overhead shall be noted to ensure the business feasibility while considering this option.

#### Focus on Large and Medium Companies for Revenue Stability

- Large companies generate most of the deal volume (~69% of total deals), therefore the company must maintain the enterprise-focused sales strategy such as offering volume discounts or developing long-term contracts with these accounts. It must also be noted that small companies have higher win rates, thus the sales company can keep a close eye on these opportunities while also helping both the sales company and the client company grow together in a win-win situation.

#### Develop Product Optimization Strategies

- Products with high sales volume shall be pushed more aggressively to potential customers whereas the pricing for low-performing products should be reevaluated to grow its sales. Promotions such as bundle sales could be pushed to increase deal attraction.

#### Monitor Deals with Long Duration Period

- Most deals close within 75-100 days, and seemingly sees less success in periods after 100 days. Every sales team should review deals that hasn't closed after 100 days and sales agents shall report to their manager and reassess the probability of closing the deal for a decision whether to follow up with the client or disengage from the deal.

#### Strengthen Relationship with Top Accounts

- The sales company should assign dedicated account managers to top cilents to build trust among between parties. Additionally, retention strategies should be implemented as well as plans to cross-sell or upsell products.

#### Implement Knowledge Sharing to Other Agents by Top Selling Agents

- The top performer Darcel Schlecht from Melvin Marxen's team should be analyzed to identify his practices or strategies so that internal training sessions can be conducted across the sales company and standardizing these strategies to improve the sales pipeline efficiency as a whole. 

#### Address Performance Imbalance

- Although Melvin Marxen (Central) is the top performing team in terms of sales, the team performance relies heavily on one standout performer named Darcel Schlecht. Therefore, it must be determined whether resources are being disproportionately allocated to this agent, which could cause unfairness and team conflict if true. Melvin Marxen's performance as a sales manager shall also be evaluated whether if he is suitable to be the manager of this team, as relying on a single sales agent could indicate the team operates on individual efforts rather than as a team.

#### Expand Prospecting Stage Across All Regions

- It was found out that only the Central region performs prospecting to generate new leads. Therefore, this stage should be introduced to both the West and East regions for better sales growth and setting prospecting KPIs across all teams.

#### Prioritizing High-Value Opportunities

- It was observed that high-value deals that are valued above `$20,000` can close quickly even within 1-2 weeks after the engage date. Therefore, deal scoring shall be implemented based on potential value, account company and market demand in order to evaluate the winning rate for these deals. Senior resources shall also be allocated to these high-value deals to maximize the success rate for these opportunities.
