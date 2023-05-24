import pandas as pd
import numpy as np
from olist.data import Olist
from olist.order import Order


class Seller:

    def __init__(self):
        # Import only data once
        olist = Olist()
        self.data = olist.get_data()
        self.matching_table = olist.get_matching_table()
        self.order = Order()

    def get_seller_features(self):
        """
        Returns a DataFrame with:
        'seller_id', 'seller_city', 'seller_state'
        """
        sellers = self.data['sellers'].copy() # Make a copy before using inplace=True so as to avoid modifying self.data
        sellers.drop('seller_zip_code_prefix', axis=1, inplace=True)
        sellers.drop_duplicates(inplace=True) # There is multiple rows per seller
        return sellers

    def get_seller_delay_wait_time(self):
        """
        Returns a DataFrame with:
        'seller_id', 'delay_to_carrier', 'wait_time'
        """
        # Get data
        order_items = self.data['order_items'].copy()
        orders = self.data['orders'].query("order_status=='delivered'").copy()

        ship = order_items.merge(orders, on='order_id')

        # Handle datetime
        ship.loc[:, 'shipping_limit_date'] = pd.to_datetime(ship['shipping_limit_date'])
        ship.loc[:, 'order_delivered_carrier_date'] = pd.to_datetime(ship['order_delivered_carrier_date'])
        ship.loc[:, 'order_delivered_customer_date'] = pd.to_datetime(ship['order_delivered_customer_date'])
        ship.loc[:, 'order_purchase_timestamp'] = pd.to_datetime(ship['order_purchase_timestamp'])

        # Compute delay and wait_time
        def delay_to_logistic_partner(d):
            days = np.mean(
                (d.shipping_limit_date - d.order_delivered_carrier_date)/np.timedelta64(24, 'h')
                )
            if days < 0:
                return abs(days)
            else:
                return 0

        def order_wait_time(d):
            days = np.mean(
                (d.order_delivered_customer_date - d.order_purchase_timestamp)/np.timedelta64(24, 'h')
                )
            return days

        delay = ship.groupby('seller_id')\
                    .apply(delay_to_logistic_partner)\
                    .reset_index()
        delay.columns = ['seller_id', 'delay_to_carrier']

        wait = ship.groupby('seller_id')\
                   .apply(order_wait_time)\
                   .reset_index()
        wait.columns = ['seller_id', 'wait_time']

        df = delay.merge(wait, on='seller_id')

        return df

    def get_active_dates(self):
        """
        Returns a DataFrame with: 'seller_id', 'date_first_sale', 'date_last_sale'
        """

        orders = self.data['orders'][['order_id', 'order_approved_at']].copy()

        # create two new columns in view of aggregating
        orders.loc[:,'date_first_sale'] = pd.to_datetime(orders['order_approved_at'])
        orders['date_last_sale'] = orders['date_first_sale']

        df = orders.merge(self.matching_table[['seller_id', 'order_id']], on="order_id")\
            .groupby('seller_id')\
            .agg({
            "date_first_sale": min,
            "date_last_sale": max
        })

        df['months_on_olist'] = round((df['date_last_sale']-df['date_first_sale']) / np.timedelta64(1, 'M'))

        return df

    def get_review_score(self):
        """
        Returns a DataFrame with:
        'seller_id', 'share_of_five_stars', 'share_of_one_stars', 'review_score', 'cost_of_reviews'
        """
        matching_table = self.matching_table
        orders_reviews = self.order.get_review_score()

        # Since same seller can appear multiple times in the same order, create a (seller <> order) matching table
        matching_table = matching_table[['order_id', 'seller_id']].drop_duplicates()
        df = matching_table.merge(orders_reviews, on='order_id')

        df['cost_of_review'] = df.review_score.map({
            1: 100,
            2: 50,
            3: 40,
            4: 0,
            5: 0
        })

        df = df.groupby('seller_id',
                        as_index=False).agg({'dim_is_one_star': 'mean',
                                             'dim_is_five_star': 'mean',
                                             'review_score': 'mean',
                                             'cost_of_review': 'sum'})
        df.columns = ['seller_id', 'share_of_one_stars',
                      'share_of_five_stars', 'review_score', 'cost_of_reviews']

        return df

    def get_quantity(self):
        """
        Returns a DataFrame with:
        'seller_id', 'n_orders', 'quantity', 'quantity_per_order'
        """
        matching_table = self.matching_table

        n_orders = matching_table.groupby('seller_id')['order_id']\
            .nunique()\
            .reset_index()
        n_orders.columns = ['seller_id', 'n_orders']

        quantity = matching_table.groupby('seller_id', as_index=False).agg({'order_id': 'count'})
        quantity.columns = ['seller_id', 'quantity']

        result = n_orders.merge(quantity, on='seller_id')
        result['quantity_per_order'] = result['quantity'] / result['n_orders']
        return result

    def get_sales(self):
        """
        Returns a DataFrame with:
        'seller_id', 'sales'
        """
        return self.data['order_items'][['seller_id', 'price']]\
            .groupby('seller_id')\
            .sum()\
            .rename(columns={'price': 'sales'})


    def get_training_data(self):
        """
        Returns a DataFrame with:
        'seller_id', 'seller_state', 'seller_city', 'delay_to_carrier',
        'wait_time', 'share_of_five_stars', 'share_of_one_stars',
        'seller_review_score', 'n_orders', 'quantity,' 'date_first_sale', 'date_last_sale', 'sales'
        """

        training_set =\
            self.get_seller_features()\
                .merge(
                self.get_seller_delay_wait_time(), on='seller_id'
               ).merge(
                self.get_active_dates(), on='seller_id'
               ).merge(
                self.get_review_score(), on='seller_id'
               ).merge(
                self.get_quantity(), on='seller_id'
               ).merge(
                self.get_sales(), on='seller_id'
               )

        # Add seller economics (revenues, profits)
        olist_monthly_fee = 80
        olist_sales_cut = 0.1

        training_set['revenues'] = training_set['months_on_olist'] * olist_monthly_fee\
            + olist_sales_cut * training_set['sales']

        training_set['profits'] = training_set['revenues'] - training_set['cost_of_reviews']

        return training_set
