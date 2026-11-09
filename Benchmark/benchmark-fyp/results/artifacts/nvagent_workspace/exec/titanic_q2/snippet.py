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
SELECT sex, COUNT(*) AS "count_*", survived FROM titanic GROUP BY sex, survived
'''
df = con.execute(sql).fetchdf()


df.columns = ['sex', 'count_*', 'survived']


# print("Columns in the dataframe:", df.columns)






fig,ax = plt.subplots(1,1,figsize=(10,4))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
df_pivot = df.pivot(index='sex', columns='survived', values='count_*')
df_pivot.plot(kind='bar', stacked=True, ax=ax, alpha=0.8)
ax.set_xlabel('sex')
ax.set_ylabel('count_*')
ax.set_title('Stacked Bar Chart of count_* by sex and survived')
ax.legend(title='survived', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()


# Print data
x_data = [df.index.tolist()]
y_data = [df[col].tolist() for col in df.columns]
print("x_data:", x_data)
print("y_data:", y_data)
print("groups:", df.columns.tolist())

plt.show()