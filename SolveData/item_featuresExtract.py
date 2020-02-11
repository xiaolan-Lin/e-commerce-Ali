# 用户特征提取(用户浏览量u_view\用户收藏量u_collect\用户加购物车量u_cart\用户购买量u_buy\
# 用户浏览到购买的转化率u_v2b\用户收藏到购买的转化率u_clt2b\用户加购物车到购买的转化率u_cart2b)

import pandas as pd
import numpy as np
# 根据相同的字段名进行外连接


def merge(feature1, feature2):
    feature = pd.merge(feature1, feature2, on='item_id', how='outer').fillna(0)
    return feature



# 采用块读取功能，读取csv文件
path1 = "./data\\tianchi_mobile_recommend_train_user.csv"

file1 = open(path1)

user_data = pd.read_csv(path1, sep=',', iterator=True)

loop = True
chunkSize = 1000
chunks1 = []
while loop:
    try:
        chunk1 = user_data.get_chunk(chunkSize)
        chunks1.append(chunk1)
    except StopIteration:
        loop = False
user_data = pd.concat(chunks1, ignore_index=True)

# len(user_data)

del user_data['user_geohash']  # 由于用户浏览商品信息时的地理位置与用户行为无关，所以将user_geohash这列去掉
del user_data['time']  # 时间也与后续分析无关，也将其删除

# 全部商品
item_id = user_data['item_id'].value_counts()
item_id = pd.DataFrame({'item_id': item_id.index})
# 商品被浏览量(并非包含全部商品，不是所有商品都被浏览过才会进行别的用户行为操作)
item_view = user_data[user_data.behavior_type == 1]['item_id'].value_counts()
i_view = pd.DataFrame({'item_id': item_view.index, 'i_view': item_view.values})
# 商品被收藏量
item_collect = user_data[user_data.behavior_type == 2]['item_id'].value_counts()
i_collect = pd.DataFrame({'item_id': item_collect.index, 'i_collect': item_collect.values})
# 商品被加购物车
item_cart = user_data[user_data.behavior_type == 3]['item_id'].value_counts()
i_cart = pd.DataFrame({'item_id': item_cart.index, 'i_cart': item_cart.values})
# 商品被购买量
item_buy = user_data[user_data.behavior_type == 4]['item_id'].value_counts()
i_buy = pd.DataFrame({'item_id': item_buy.index, 'i_buy': item_buy.values})
# 商品被浏览到被购买的转化率
item_v2b = merge(i_view, i_buy)
i_v2b = merge(item_id, item_v2b)
np.seterr(invalid='ignore')
i_v2b['i_v2b'] = i_v2b.apply(lambda x: x[2] / x[1], axis=1).replace(np.inf, 0).fillna(0)
i_v2b['i_v2b'] = round(i_v2b['i_v2b'], 2)
# 商品被收藏到被购买的转化率
i_clt2b = merge(i_v2b, i_collect)
i_clt2b['i_clt2b'] = i_clt2b.apply(lambda x: x[2] / x[4], axis=1).replace(np.inf, 0).fillna(0)
i_clt2b['i_clt2b'] = round(i_clt2b['i_clt2b'], 2)
# 商品被加购物车到被购买的转化率
i_cart2b = merge(i_clt2b, i_cart)
i_cart2b['i_cart2b'] = i_cart2b.apply(lambda x: x[2] / x[6], axis=1).replace(np.inf, 0).fillna(0)
i_cart2b['i_cart2b'] = round(i_cart2b['i_cart2b'], 2)
# 存放提取后的商品行为特征集
i_cart2b.to_csv('./after_data\\item_features.csv', index=False)



df = pd.DataFrame()
