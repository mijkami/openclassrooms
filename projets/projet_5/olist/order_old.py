import os
import pandas as pd
import numpy as np
from olist.utils import haversine_distance
from olist.data import Olist


class Order:
    '''
    DataFrames containing all orders as index,
    and various properties of these orders as columns
    '''

    def __init__(self):
        # Assign an attribute ".data" to all new instances of Order
        self.data = Olist().get_data()

    def get_wait_time(self, is_delivered=True):
        """
        02-01 > Returns a DataFrame with:
        [order_id, wait_time, expected_wait_time, delay_vs_expected, order_status]
        and filtering out non-delivered orders unless specified
        """
        # Hint: Within this instance method, you have access to the instance of the class Order in the variable self, as well as all its attributes
        orders = self.data['orders'].copy() # good practice to be sure not to modify your `data` variable
        orders.sort_values('order_delivered_customer_date')
        orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
        orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])
        orders['wait_time'] = (orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']).dt.days
        orders['expected_wait_time'] = (orders['order_estimated_delivery_date'] - orders['order_purchase_timestamp']).dt.days
        orders['delay_vs_expected']  = orders['wait_time'] - orders['expected_wait_time']

        def positive_value(d):
            return d if d > 0 else 0

        orders['delay_vs_expected'] = orders['delay_vs_expected'].map(positive_value)
        orders = orders[['order_id', 'wait_time', 'expected_wait_time', 'delay_vs_expected', 'order_status']]
        return orders

    def get_review_score(self):
        """
        02-01 > Returns a DataFrame with:
        order_id, dim_is_five_star, dim_is_one_star, review_score
        """
        reviews = self.data['order_reviews'].copy()
        # dim_is_five_star (int) 1 if the order received a five_star, 0 otherwise
        def dim_five_star(d):
            return 1 if d == 5 else 0

        # dim_is_one_star (int) 1 if the order received a one_star, 0 otherwise
        def dim_one_star(d):
            return 1 if d == 1 else 0

        reviews["dim_is_five_star"] = reviews["review_score"].map(dim_five_star) # --> Series([0, 1, 1, 0, 0, 1 ...])
        reviews["dim_is_one_star"] = reviews["review_score"].map(dim_one_star) # --> Series([0, 1, 1, 0, 0, 1 ...])
        reviews = reviews[['order_id', 'dim_is_five_star', 'dim_is_one_star', 'review_score']]
        return reviews

    def get_number_products(self):
        """
        02-01 > Returns a DataFrame with:
        order_id, number_of_products
        """

    def get_number_sellers(self):
        """
        02-01 > Returns a DataFrame with:
        order_id, number_of_sellers
        """

    def get_price_and_freight(self):
        """
        02-01 > Returns a DataFrame with:
        order_id, price, freight_value
        """

    def get_distance_seller_customer(self):
        """
        02-01 > Returns a DataFrame with order_id
        and distance between seller and customer
        """
        # Optional
        # Hint: you can use the haversine_distance logic coded in olist/utils.py

    def get_training_data(self, is_delivered=True,
                          with_distance_seller_customer=False):
        """
        02-01 > Returns a clean DataFrame (without NaN), with the following columns:
        [order_id, wait_time, expected_wait_time, delay_vs_expected, order_status,
        dim_is_five_star, dim_is_one_star, review_score, number_of_products,
        number_of_sellers, price, freight_value, distance_customer_seller]
        """
        # Hint: make sure to re-use your instance methods defined above
