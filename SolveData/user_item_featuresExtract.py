# 用户-商品行为特征提取
#
import pandas as pd


# 根据用户id、商品id进行外连接


def merge(feature1, feature2):
    feature = pd.merge(feature1, feature2, on=['user_id', 'item_id'], how='outer').fillna(0)
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

del user_data['user_geohash']  # 地理位置无关，将其删除

# 该用户对该商品浏览量
ui_view = pd.DataFrame(user_data[user_data.behavior_type == 1].groupby(['user_id', 'item_id']).size(),
                       columns=['ui_view'])
ui_view.reset_index(inplace=True)
# 该用户是否收藏该商品
ui_collect = pd.DataFrame(user_data[user_data.behavior_type == 2].groupby(['user_id', 'item_id']).size(),
                          columns=['ui_collect'])
ui_collect.reset_index(inplace=True)
# 该用户收藏过该商品用1表示，否则用0
ui_collect['ui_collect'] = 1
ui_collect = merge(ui_view, ui_collect)
# 该用户是否将该商品加入过购物车
ui_cart = pd.DataFrame(user_data[user_data.behavior_type == 3].groupby(['user_id', 'item_id']).size(),
                       columns=['ui_cart'])
ui_cart.reset_index(inplace=True)
# 该用户加购过该商品用1表示，否则用0
ui_cart['ui_cart'] = 1
ui_cart = merge(ui_collect, ui_cart)
# 该用户对该商品的历史购买量
ui_buy = pd.DataFrame(user_data[user_data.behavior_type == 4].groupby(['user_id', 'item_id']).size(),
                      columns=['ui_buy'])
ui_buy.reset_index(inplace=True)
ui_buy = merge(ui_cart, ui_buy)

# 首先提取每样商品首次出现的时间
first_t = pd.DataFrame(user_data.groupby(['item_id', 'time']).size())
first_t.reset_index(inplace=True)
del first_t[0]
first_t.rename(columns={'time': 'first_appear_time'}, inplace=True)
# 取groupby分组后第一个（时间最早）
first_t = first_t.groupby(['item_id']).head(1)
# 合并后全部商品拥有最早被浏览的时间
ui_features = pd.merge(ui_buy, first_t, on='item_id', how='outer')
ui_time = pd.DataFrame(user_data.groupby(['user_id', 'item_id', 'time']).size())
ui_time.reset_index(inplace=True)
del ui_time[0]
# 用户首次与商品发生交互的时间
ui_first_time = ui_time.groupby(['user_id', 'item_id']).head(1)
ui_first_time.rename(columns={'time': 'ui_first_time'}, inplace=True)
ui_features = merge(ui_features, ui_first_time)
ui_last_time = ui_time.groupby(['user_id', 'item_id']).tail(1)
ui_last_time.rename(columns={'time': 'ui_last_time'}, inplace=True)
ui_features = merge(ui_features, ui_last_time)
# 商品首次出现交互距今时长
ui_features['iu_first_t'] = pd.to_timedelta(
    pd.to_datetime(ui_features['ui_last_time']) - pd.to_datetime(ui_features['first_appear_time'])).dt.days
# 用户首次与商品交互距今时长
ui_features['ui_first_t'] = pd.to_timedelta(
    pd.to_datetime(ui_features['ui_last_time']) - pd.to_datetime(ui_features['ui_first_time'])).dt.days
# 删除first_appear_time、ui_first_time、ui_last_time三列
ui_features = ui_features.drop(['first_appear_time', 'ui_first_time', 'ui_last_time'], axis=1)
ui_features.to_csv('./after_data\\user_item_interactive.csv', index=False)

test = pd.DataFrame({'item_id': [10012, 10013, 10014, 10012], 'be': [32, 83, 546, 32], 'sum': [29, 32, 134, 45]})
test1 = pd.DataFrame(test.groupby(['item_id', 'be', 'sum']).size())
test1.reset_index(inplace=True)
test1


