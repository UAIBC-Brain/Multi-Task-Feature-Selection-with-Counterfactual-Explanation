# import DiCE
import dice_ml # 导入DiCE库，它是一个用于生成反事实解释的python库
from dice_ml.utils import helpers  # 导入Dice库的辅助函数，这些函数用于辅助数据加载和模型训练
from sklearn.compose import ColumnTransformer # 从scikit—learn库导入ColumnTransformer，用于处理数据的列转换
from sklearn.model_selection import train_test_split # 从scikit-learn库导入train_test_split函数，将数据集分为训练集和测试集
from sklearn.pipeline import Pipeline # 从scikit-learn库导入Pipeline，用于创建数据预处理和模型训练的流水线
from sklearn.preprocessing import StandardScaler, OneHotEncoder # 从scikit-learn库导入StandardScaler和OneHotEncoder，用于数值特征的标准化和分类特征的独热编码
from sklearn.ensemble import RandomForestClassifier # 从scikit-learn库导入RandomForestClassifier，这是一个用于分类任务的随机森林分类器
import pandas as pd

dataset = pd.read_csv('G:\\AWSLwork\\Awsl9\\Result-noindex\\Xiangya_selected.csv')
# 1是患者，0是正常人

# 创建一个dice_ml_Data实例d，该实例包括数据集dataset，定义了连续特征列以及目标变量名称
d = dice_ml.Data(dataframe=dataset, continuous_features=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                                                         '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                                                         '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
                                                         '31', '32', '33', '34', '35', '36', '37', '38', '39', '40',
                                                         '41', '42', '43', '44', '45', '46'], outcome_name='label')

# 训练自定义ML模型
target = dataset["label"]
# Split data into train and test
datasetX = dataset.drop("label", axis=1)  # 从名为dataset的数据集中删除名为label的列
x_train, x_test, y_train, y_test = train_test_split(datasetX,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=target)

numerical = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
             "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
             "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
             "31", "32", "33", "34", "35", "36", "37", "38", "39", "40",
             "41", "42", "43", "44", "45", "46"]
categorical = x_train.columns.difference(numerical)

# We create the preprocessing pipelines for both numeric and categorical data.我们为数字和分类数据创建预处理管道
# numeric_transformer和categorical_transformer是用于数值特征和分类特征的预处理管道
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
# transformations是一个ColumnTransformer,将数值特征和分类特征的预处理管道组合在一起
transformations = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical),
        ('cat', categorical_transformer, categorical)])

# clf是一个包含数据预处理和随机森林分类器的完整预测管道
clf = Pipeline(steps=[('preprocessor', transformations),
                      ('classifier', RandomForestClassifier())])
model = clf.fit(x_train, y_train)

# 将经过训练的ML模型提供给DiCE的模型对象
backend = 'sklearn' # 指定DiCE模型的后端为scikit_learn
m = dice_ml.Model(model=model, backend=backend) # 创建一个DiCE模型对象m，并将训练好的自定义ML模型与其关联

# ********************【1.（原方法）生成多样性反事实解释】**********************
# initiate DiCE
exp_random = dice_ml.Dice(d, m, method="random") # exp_random是一个DiCE解释实例，使用了数据d和模型m，并且使用random方法来生成反事实解释
indices_label_0 = y_train[y_train == 0].index # 筛选标签为0的行索引
query_instances = x_train.loc[indices_label_0] # query_instances是要生成反事实解释的查询实例，选择x_train中标签为0的样本

# query_instances = x_train[160:165] # query_instances是要生成反事实解释的查询实例，选择x_train中的一部分样本
# dice_exp_random用于生成反事实解释，包括生成与desired_class相反的反事实解释
dice_exp_random = exp_random.generate_counterfactuals(query_instances, total_CFs=10, desired_class=1, verbose=False)  # random_seed=9控制可随机性，total_CFs用来控制生成反事实的数量

# 可视化
dice_exp_random.visualize_as_dataframe(show_only_changes=True)

for i, cf_example in enumerate(dice_exp_random.cf_examples_list):
    file_path = r'G:\AWSLwork\Awsl9\5Xiangya\counterfactuals_{}.csv'.format(i)
    cf_example.final_cfs_df.to_csv(file_path, index=False)

# ********************【2.筛选x_train中标签为0的数据】**********************
# initiate DiCE
# exp_random = dice_ml.Dice(d, m, method="random") # exp_random是一个DiCE解释实例，使用了数据d和模型m，并且使用random方法来生成反事实解释
# indices_label_0 = y_train[y_train == 0].index # 筛选标签为0的行索引
# query_instances = x_train.loc[indices_label_0] # query_instances是要生成反事实解释的查询实例，选择x_train中标签为0的样本
# 【-------------------仅用来保存x_train在原文件中对应索引（用一次即可）-----------------】
# 合并 query_instances 和对应的行索引  （行索引+2=原文件中行索引）
query_instances_with_index = query_instances.copy()
query_instances_with_index['index'] = query_instances_with_index.index
# 保存到 CSV 文件
query_instances_with_index.to_csv('Aquery_instances_with_index+2.csv', index=False)
# 【-------------------仅用来保存x_train在原文件中对应索引（用一次即可）-----------------】
# 可视化
# dice_exp_random.visualize_as_dataframe(show_only_changes=True)
# for i, cf_example in enumerate(dice_exp_random.cf_examples_list):
#     file_path = r'D:\2\py_project1\反事实解释结果保存\counterfactuals_{}.csv'.format(i)
#     cf_example.final_cfs_df.to_csv(file_path, index=False)

# ********************【保存方法】**********************
# 1.循环保存生成的反事实
# for i, cf_example in enumerate(dice_exp_random.cf_examples_list):
#     file_path = r'G:\AWSLwork\Awsl9\1CORE_CF\counterfactuals_{}.csv'.format(i)
#     cf_example.final_cfs_df.to_csv(file_path, index=False)
# # 2.一般方法保存生成的反事实示例
# dice_exp_random.cf_examples_list[0].final_cfs_df.to_csv(path_or_buf='counterfactuals0.csv', index=False)
