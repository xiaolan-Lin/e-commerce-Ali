# 商品特征提取(商品被浏览量i_view\商品被收藏量i_collect\商品被加购物车量i_cart\商品被购买量i_buy\商品被浏览到被购买的转化率i_v2b\
# 商品被收藏到被购买的转化率i_clt2b\商品被加购物车到被购买的转化率i_cart2b)

# behavior_type列的特别说明：
# a. click（点击浏览）: 1
# b. collect（收藏）: 2
# c. add-to-cart（添加到购物车）: 3
# d. payment（下单）: 4


import pandas as pd
import numpy as np
# 根据相同的字段名进行外连接


def merge(feature1, feature2):
    feature = pd.merge(feature1, feature2, on='user_id', how='outer').fillna(0)
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

len(user_data)

del user_data['user_geohash']  # 由于用户浏览商品信息时的地理位置与用户行为无关，所以将user_geohash这列去掉
del user_data['time']  # 时间也与后续分析无关，也将其删除

# 全部用户
user_id = user_data['user_id'].value_counts()
user_id = pd.DataFrame({'user_id': user_id.index})
# 用户浏览量
user_view = user_data[user_data.behavior_type == 1]['user_id'].value_counts()
u_view = pd.DataFrame({'user_id': user_view.index, 'u_view': user_view.values})
# 用户收藏量
user_collect = user_data[user_data.behavior_type == 2]['user_id'].value_counts()
u_collect = pd.DataFrame({'user_id': user_collect.index, 'u_collect': user_collect.values})
# 用户加购物车量
user_cart = user_data[user_data.behavior_type == 3]['user_id'].value_counts()
u_cart = pd.DataFrame({'user_id': user_cart.index, 'u_cart': user_cart.values})
# 用户购买量
user_buy = user_data[user_data.behavior_type == 4]['user_id'].value_counts()
u_buy = pd.DataFrame({'user_id': user_buy.index, 'u_buy': user_buy.values})
# 用户浏览到购买的转化率
user_v2b = merge(u_view, u_buy)
u_v2b = merge(user_id, user_v2b)
np.seterr(invalid='ignore')
u_v2b['u_v2b'] = u_v2b.apply(lambda x: x[2] / x[1], axis=1).replace(np.inf, 0).fillna(0)
u_v2b['u_v2b'] = round(u_v2b['u_v2b'], 2)
# 用户收藏到购买的转化率
u_clt2b = merge(u_v2b, u_collect)
u_clt2b['u_clt2b'] = u_clt2b.apply(lambda x: x[2] / x[4], axis=1).replace(np.inf, 0).fillna(0)
u_clt2b['u_clt2b'] = round(u_clt2b['u_clt2b'], 2)
# 用户加购物车到购买的转化率
u_cart2b = merge(u_clt2b, u_cart)
u_cart2b['u_cart2b'] = u_cart2b.apply(lambda x: x[2] / x[6], axis=1).replace(np.inf, 0).fillna(0)
u_cart2b['u_cart2b'] = round(u_cart2b['u_cart2b'], 2)
# 存放同去后的用户行为特征集
u_cart2b.to_csv('./after_data\\user_features.csv', index=False)


