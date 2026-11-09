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
SELECT alcohol, quality FROM wine_quality
'''
df = con.execute(sql).fetchdf()
con.close()

# rename columns
df.columns = ['alcohol','quality']


# print("Columns in the dataframe:", df.columns)






fig,ax = plt.subplots(1,1,figsize=(10,4))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


ax.scatter(df['alcohol'], df['quality'])

ax.set_xlabel('alcohol')
ax.set_ylabel('quality')
ax.set_title(f'SCATTER Chart of quality by alcohol')
plt.xticks(rotation=45)
plt.tight_layout()


# Print data
print("x_data:", df['alcohol'].tolist())
print("y_data:", df['quality'].tolist())

plt.show()