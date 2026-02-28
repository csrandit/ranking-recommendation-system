from abc import ABC, abstractmethod

class BaseRecommender(ABC):

    @abstractmethod
    def fit(self, train_df):
        pass

    @abstractmethod
    def recommend(self, user_id, k=10):
        pass