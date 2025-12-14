from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans


class Preprocess_catboost(BaseEstimator, TransformerMixin):
    gender_mapping = {1: 1, 0: 0}

    def __init__(self):
        return

    def fit(self, X, y=None):
        X_train = X.copy()

        # Добавление Age_square
        self.X_train_age_mean = X_train['Age'].mean()
        X_train['Age_square'] = (X_train['Age'] - self.X_train_age_mean) ** 2

        # print("TRAIN: Gender", X_train['Gender'])
        # print("TRAIN: Gender_Code", X_train['Gender'].map(self.gender_mapping))
        #
        # # Переводим пол в числа
        # X_train['Gender_Code'] = X_train['Gender'].map(self.gender_mapping)

        X_train['Gender_Code'] = X_train['Gender']

        # Кластеризуем регионы с учетом каналов продаж
        # Группируем по Region_Code и Policy_Sales_Channel
        region_stats = X_train.groupby(['Region_Code', 'Policy_Sales_Channel']).agg({
            'Gender_Code': 'mean',
            'Driving_License': 'mean',
            'Annual_Premium': 'mean',
            'Vintage': 'mean',
            'Age': 'mean',
            'Age_square': 'mean'
        }).reset_index()

        self.kmeans_region = KMeans(n_clusters=7, random_state=42)
        self.kmeans_region = self.kmeans_region.fit(region_stats.drop(['Region_Code', 'Policy_Sales_Channel'], axis=1))

        return self

    def transform(self, X):
        X_test = X.copy()
        int_columns = ['Region_Code', 'Policy_Sales_Channel']
        for col_name in int_columns:
            X_test[col_name] = X_test[col_name].astype('int64')

        X_test['Age_square'] = (X_test['Age'] - self.X_train_age_mean) ** 2

        # print("TEST: Gender", X_test['Gender'])
        # print("TEST: Gender_Code", X_test['Gender'].map(self.gender_mapping))
        #
        # X_test['Gender_Code'] = X_test['Gender'].map(self.gender_mapping)

        X_test['Gender_Code'] = X_test['Gender']

        region_stats_test = X_test.groupby(['Region_Code', 'Policy_Sales_Channel']).agg({
            'Gender_Code': 'mean',
            'Driving_License': 'mean',
            'Annual_Premium': 'mean',
            'Vintage': 'mean',
            'Age': 'mean',
            'Age_square': 'mean'
        }).reset_index()

        region_stats_test['Region_Cluster'] = self.kmeans_region.predict(
            region_stats_test.drop(['Region_Code', 'Policy_Sales_Channel'], axis=1))
        region_cluster_map_test = region_stats_test.set_index(['Region_Code', 'Policy_Sales_Channel'])[
            'Region_Cluster'].to_dict()
        X_test['Region_Cluster'] = X_test.set_index(['Region_Code', 'Policy_Sales_Channel']).index.map(
            region_cluster_map_test)

        X_test.drop(['Gender_Code'], axis=1, inplace=True)

        return X_test
