import seaborn as sns
import polars as pl
from matplotlib import pyplot as plt

print("Loading Plot")
df: pl.DataFrame = pl.read_csv("data.csv")

print("Aggregating Data")
df_agg = df.group_by(['model', 'iter']).agg(pl.col('evaluation').mean()).filter(pl.col('iter').mod(1000).eq(0))

print("Plotting")
sns.lineplot(data=df, x='iter', y='evaluation', hue='model')

print("Saving")
plt.savefig("images/evaluation.png")
