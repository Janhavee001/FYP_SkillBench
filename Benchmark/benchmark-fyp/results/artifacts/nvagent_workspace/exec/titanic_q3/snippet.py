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
SELECT survived, AVG(age) AS avg_age FROM titanic WHERE NOT age IS NULL GROUP BY survived
'''
df = con.execute(sql).fetchdf()
con.close()

# rename columns
df.columns = ['survived','avg_age']


# print("Columns in the dataframe:", df.columns)






fig,ax = plt.subplots(1,1,figsize=(10,4))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.bar(df['survived'], df['avg_age'])



ax.set_xlabel('survived')
ax.set_ylabel('avg_age')
ax.set_title(f'BAR Chart of avg_age by survived')
plt.xticks(rotation=45)
plt.tight_layout()


# Print data
print("x_data:", df['survived'].tolist())
print("y_data:", df['avg_age'].tolist())

plt.show()