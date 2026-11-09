import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import pandas as pd
import os
import duckdb


data_folder = 'C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/results/artifacts/nvagent_workspace/databases/wine_quality'


con = duckdb.connect(database=':memory:')


csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
for file in csv_files:
    table_name = os.path.splitext(file)[0]
    con.execute(f"CREATE VIEW {table_name} AS SELECT * FROM read_csv_auto('{os.path.join(data_folder, file)}')")


sql = f'''
SELECT CASE WHEN quality >= 7 THEN 'High' WHEN quality <= 4 THEN 'Low' END AS quality_group, AVG(alcohol) AS avg_alcohol FROM wine_quality WHERE quality >= 7 OR quality <= 4 GROUP BY 1
'''
df = con.execute(sql).fetchdf()
con.close()

# rename columns
df.columns = ['quality_group','avg_alcohol']


# print("Columns in the dataframe:", df.columns)






fig,ax = plt.subplots(1,1,figsize=(10,4))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.bar(df['quality_group'], df['avg_alcohol'])



ax.set_xlabel('quality_group')
ax.set_ylabel('avg_alcohol')
ax.set_title(f'BAR Chart of avg_alcohol by quality_group')
plt.xticks(rotation=45)
plt.tight_layout()


# Print data
print("x_data:", df['quality_group'].tolist())
print("y_data:", df['avg_alcohol'].tolist())

plt.show()