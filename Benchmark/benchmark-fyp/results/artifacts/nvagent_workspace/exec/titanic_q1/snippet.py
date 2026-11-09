import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import pandas as pd
import os
import duckdb


data_folder = 'C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/results/artifacts/nvagent_workspace/databases/titanic'


con = duckdb.connect(database=':memory:')


csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
for file in csv_files:
    table_name = os.path.splitext(file)[0]
    con.execute(f"CREATE VIEW {table_name} AS SELECT * FROM read_csv_auto('{os.path.join(data_folder, file)}')")


sql = f'''
SELECT sex, AVG(survived) AS avg_survived FROM titanic GROUP BY sex
'''
df = con.execute(sql).fetchdf()
con.close()

# rename columns
df.columns = ['sex','avg_survived']


# print("Columns in the dataframe:", df.columns)






fig,ax = plt.subplots(1,1,figsize=(10,4))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.bar(df['sex'], df['avg_survived'])



ax.set_xlabel('sex')
ax.set_ylabel('avg_survived')
ax.set_title(f'BAR Chart of avg_survived by sex')
plt.xticks(rotation=45)
plt.tight_layout()


# Print data
print("x_data:", df['sex'].tolist())
print("y_data:", df['avg_survived'].tolist())

plt.show()