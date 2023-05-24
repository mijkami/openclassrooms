import pandas as pd
import numpy as np
import math
from olist.data import Olist
from olist.order import Order


class Review:

    def __init__(self):
        # Import only data once
        olist = Olist()
        self.data = olist.get_data()
        self.matching_table = olist.get_matching_table()
        self.order = Order()

    def get_review_length(self):
        """
        04-02 > Returns a DataFrame with:
       'review_id', 'length_review', 'review_score'
        """
        reviews = self.data['order_reviews']

        def length_comment(d):
            if type(d) is str:
                return len(d)
            else:
                return 0

        reviews.loc[:, 'length_review'] = \
            reviews['review_comment_message'].apply(length_comment)

        return reviews[['review_id', 'length_review',
                        'review_score']]

    def get_main_product_category(self):
        """
        04-02 > Returns a DataFrame with:
       'review_id', 'order_id','product_category_name'
        """
        products = self.data['products']
        order_items = self.data['order_items']
        reviews = self.data['order_reviews']

        # Groupby category, sort and drop_duplicates
        order_product_category = \
            order_items.merge(products[['product_category_name',
                                        'product_id']],
                              on='product_id')\
                       .groupby(['order_id', 'product_category_name'],
                                as_index=False)\
                       .agg({'price': 'sum'}).sort_values(by=['order_id',
                                                              'price'],
                                                          ascending=False)\
                       .drop_duplicates(['order_id'])\
                       .drop(['price'], axis=1)

        return reviews.merge(order_product_category,
                             on='order_id')[['review_id',
                                             'order_id',
                                             'product_category_name']]

    def get_training_data(self):

        df = self.get_review_length()\
                 .merge(self.get_main_product_category(),
                        on='review_id')
        return df
