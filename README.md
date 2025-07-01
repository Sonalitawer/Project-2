# Project-2
Heart attack prediction
<br>

import numpy as np
<br>
import pandas as pd
<br>
import matplotlib.pyplot as plt
<br>
import seaborn as sns
<br>

df = pd.read_csv("1.healthcare-dataset-stroke-data.csv")
<br>

df.head(5)
<br>

df1 = df.drop(["id"],axis=1)
<br>
