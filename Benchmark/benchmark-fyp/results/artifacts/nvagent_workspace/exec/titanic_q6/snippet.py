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
SELECT age, fare FROM titanic
'''
df = con.execute(sql).fetchdf()
con.close()

# rename columns
df.columns = ['age','fare']


# print("Columns in the dataframe:", df.columns)






fig,ax = plt.subplots(1,1,figsize=(10,4))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


ax.scatter(df['age'], df['fare'])

ax.set_xlabel('age')
ax.set_ylabel('fare')
ax.set_title(f'SCATTER Chart of fare by age')
plt.xticks(rotation=45)
plt.tight_layout()


# Print data
print("x_data:", df['age'].tolist())
print("y_data:", df['fare'].tolist())

plt.show()