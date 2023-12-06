import pandas as pd
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

data = pd.read_csv('doc2vec_bias.csv')
data = data.drop(columns='Unnamed: 0', axis=1)
# df_filtered = data[data['bias_score'] != 1]
# df_filtered['bias_score'] = df_filtered['bias_score'].replace(2, 1)

# print(df_filtered)

def str_to_array(str):
    start = str.index('[')
    end = str.index(']')
    p_nums = str[start + 1: end].replace('\n', '')
    p_vectors = list(map(lambda y: float(y), filter(lambda x: x != '', p_nums.split(' '))))
    return p_vectors

data['paragraph_vectors'] = data['paragraph_vectors'].apply(str_to_array)

# print(data)

# print(len(data['paragraph_vectors'][0]))
column_names = [str(i) for i in range(100)]
# print('COLUMN NAMES: ', column_names)

expanded_df = data.join(pd.DataFrame(data['paragraph_vectors'].tolist(), columns=column_names))
expanded_df.drop('paragraph_vectors', axis=1, inplace=True)

# print('EXPANDED DF: ', expanded_df)

data_unscaled = expanded_df[['topic', 'source', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99']]
data_scaled = scaler.fit_transform(data_unscaled.to_numpy())
data_scaled_pd = pd.DataFrame(data_scaled)

# print('DATA SCALED ', data_scaled_pd)

expanded_df['topic'] = data_scaled_pd[0]
expanded_df['source'] = data_scaled_pd[1]

# print()

for i in range(100):
    expanded_df[str(i)] = data_scaled_pd[i+2]

expanded_df.to_csv('normalized_data_WITH_CENTER.csv')