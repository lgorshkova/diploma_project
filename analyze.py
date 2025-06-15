import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

df = pd.read_csv(r"C:\Users\lizag\OneDrive\Рабочий стол\Kozminski\diplom\analzing\dataset_seo.csv")

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
binary_cols = ['Long_Title', 'Full_Bullets', 'Detailed_Bullets',
               'Many_Images', 'Has_Video', 'Has_Reviews_20+', 'High_Rating']
df[binary_cols] = df[binary_cols].applymap(lambda x: 1 if x == 'yes' else 0)

#group means comparison
grouped = df.groupby("Optimization_Group")[['Monthly Revenue', 'Monthly Sales', 
                                             'LQS_Score', 'Keyword_Coverage_Count',
                                             'Avg_Organic_Rank']].mean()
print(grouped)

opt_1 = df[df["Optimization_Group"] == 1]
opt_0 = df[df["Optimization_Group"] == 0]

t_stat, p_value = stats.ttest_ind(opt_1["Monthly Revenue"], opt_0["Monthly Revenue"])
print(f"T-test for Monthly Revenue: t = {t_stat:.2f}, p = {p_value:.4f}")

#revenue comparison
sns.boxplot(data=df, x="Optimization_Group", y="Monthly Revenue")
plt.title("Monthly Revenue by Optimization Group")
plt.xticks([0, 1], ["Non-Optimized", "Optimized"])
plt.ylim(0, 30000)
plt.show()

#rank comparison
sns.histplot(df, x="Avg_Organic_Rank", hue="Optimization_Group", kde=True)
plt.title("Distribution of Organic Rank by Optimization Status")
plt.show()


grouped_stats = df.groupby("Optimization_Group")[['Monthly Revenue', 'Monthly Sales', 
                                                   'LQS_Score', 'Keyword_Coverage_Count',
                                                   'Avg_Organic_Rank']].mean().round(2).reset_index()

fig, ax = plt.subplots(figsize=(10, 2))
ax.axis('off')
ax.axis('tight')


table = ax.table(cellText=grouped_stats.values,
                 colLabels=grouped_stats.columns,
                 cellLoc='center',
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)


plt.title("Group Statistics: Optimized vs. Non-Optimized Listings", fontsize=12)


plt.savefig("optimization_group_table.png", bbox_inches='tight')
plt.show()

#scatter
sns.lmplot(data=df, x="Keyword_Coverage_Count", y="Monthly Revenue", hue="Optimization_Group", aspect=1.5)
plt.title("Keyword Coverage vs. Monthly Revenue")
plt.show()

underperf = df[(df['Optimization_Group'] == 1) & (df['Monthly Revenue'] < 1000)]
print(underperf[['ASIN', 'Monthly Revenue', 'LQS_Score', 'Keyword_Coverage_Count']])

seo_subset = df[['Keyword_Coverage_Count', 'Avg_Organic_Rank', 'LQS_Score', 'Monthly Sales']]

#correlation matrix
correlation_matrix = seo_subset.corr().round(2)
plt.figure(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', fmt='.2f')
plt.xticks(rotation=45)
plt.title("Correlation Matrix of SEO and Performance Metrics")
plt.tight_layout()
plt.savefig("seo_kpi_correlation_matrix.png", bbox_inches='tight')
plt.show()


binary_cols = ['Long_Title', 'Full_Bullets', 'Detailed_Bullets',
               'Many_Images', 'Has_Video', 'Has_Reviews_20+', 'High_Rating']
df[binary_cols] = df[binary_cols].applymap(lambda x: 1 if x == 'yes' else 0)

#scatter with regression line
plt.figure(figsize=(8, 5))
sns.regplot(data=df, x='Keyword_Coverage_Count', y='Monthly Sales', scatter_kws={'alpha':0.6})
plt.title("Keyword Coverage vs. Monthly Sales")
plt.xlabel("Keyword Coverage Count")
plt.ylabel("Monthly Sales")
plt.grid(True)
plt.savefig("keyword_vs_sales_regression.png", bbox_inches='tight')
plt.show()
