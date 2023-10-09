import numpy as np
import pandas as pd

item2cat = pd.read_csv('ori_book_item_cate.txt',
                       delimiter=',', header=None, names=['item_id', 'cate_id'],
                       dtype={'item_id': int, 'cate_id': str})

print(item2cat.head())
item2cat.columns = ['item_id', 'cate_id']
grouped_counts = item2cat['cate_id'].value_counts()
#print(grouped_counts)
infrequent_categories = grouped_counts[grouped_counts < 50].index.tolist()
#print(infrequent_categories)
item2cat['cate_id'] = item2cat['cate_id'].replace(infrequent_categories, '-1')
print(item2cat.head())
print(len(item2cat['cate_id'].unique()))

#item2cat.to_csv('book_item_cate.txt', sep=',', index=False)
# p_num = 0

# for i in range(len(item2cat)):
#     cate_id = item2cat['cate_id'][i]
#     #item_id = item2cat['item_id'][i]
#     if cate_id in infrequent_categories:
#         item2cat['cate_id'][i] = -1
