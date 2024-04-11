from matplotlib import pyplot as plt
import polars as pl


df: pl.DataFrame = pl.read_csv("data.csv")
