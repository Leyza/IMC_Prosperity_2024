from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any
import string
import numpy as np
import jsonpickle as js
import json
import math


class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        print(json.dumps([
            self.compress_state(state),
            self.compress_orders(orders),
            conversions,
            trader_data,
            self.logs,
        ], cls=ProsperityEncoder, separators=(",", ":")))

        self.logs = ""

    def compress_state(self, state: TradingState) -> list[Any]:
        return [
            state.timestamp,
            "",
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed


logger = Logger()


class Trader:
    POSITION_LIMITS = {"AMETHYSTS": 20, "STARFRUIT": 20, "ORCHIDS": 100}
    MAX_HISTORY_LENGTH = {"AMETHYSTS": 0, "STARFRUIT": 50, "ORCHIDS": 0}
    TIMESTAMP_INTERVAL = 100

    def sma(self, price_history, history_length, curr_timestamp, pad_beginning=False, initial_avg=0):
        """
        Calculate the simple moving average on the price history.
        """
        total = 0
        count = 0

        if pad_beginning and len(price_history) > 0 and price_history[0]["timestamp"] > curr_timestamp - history_length:
            count = (price_history[0]["timestamp"] - curr_timestamp + history_length) / self.TIMESTAMP_INTERVAL
            total = initial_avg * count

        for trade in price_history:
            if trade["timestamp"] < curr_timestamp - history_length:
                continue

            total += trade["price"]
            count += 1

        return total / count
    
    def ema(self, price_history, history_length, curr_timestamp):
        """
        Calculate the exponential moving average on the price history.
        """
        total = 0
        count = 0

        for trade in price_history:
            if trade["timestamp"] < curr_timestamp - history_length:
                continue

            count += 1
            k = 2 / (count + 1)
            total = trade["price"] * k + total * (1 - k)

        return total

    def macd(self, price_history, short_length, long_length, curr_timestamp, sma=True):
        if sma:
            short = self.sma(price_history, short_length, curr_timestamp)
            long = self.sma(price_history, long_length, curr_timestamp)
        else:
            short = self.ema(price_history, short_length, curr_timestamp)
            long = self.ema(price_history, long_length, curr_timestamp)

        return short - long

    def volatility(self, price_history, history_length, curr_timestamp, mean, pad_beginning=False, initial_sqr_residual=0.0):
        """
        Calculate the historical price volatility.
        """
        total = 0
        count = 0

        if pad_beginning and len(price_history) > 0 and price_history[0]["timestamp"] > curr_timestamp - history_length:
            count = (price_history[0]["timestamp"] - curr_timestamp + history_length) / self.TIMESTAMP_INTERVAL
            total = initial_sqr_residual * count

        for trade in price_history:
            if trade["timestamp"] < curr_timestamp - history_length:
                continue

            total += (trade["price"] - mean) ** 2
            count += 1

        return np.sqrt(total / count)

    def lin_regression(self, train_x, train_y):
        """
        Apply linear regression to the given training data set. Return the coefficients and intercept.
        """
        res = np.linalg.lstsq(train_x, train_y, rcond=None)[0]
        return res[:-1], res[-1]        # coefs, intercept

    def preprocess_for_lr(self, price_history, num_vars):
        """
        Preprocess price history into the training data matrix and training targets. Num_vars determines the number of features.
        """
        result = {f"x{i}": [] for i in range(num_vars)}
        result['target'] = []

        past_vals = []
        for i in range(len(price_history)):
            past_vals.append(price_history[i]['price'])

            if i >= num_vars:
                for j in range(num_vars):
                    result[f"x{j}"].append(past_vals[j])
                result['target'].append(past_vals[-1])
                past_vals.pop(0)

        train_x = np.dstack([[result[f"x{i}"]] for i in range(num_vars)]).squeeze()
        train_x = np.c_[train_x, np.ones(train_x.shape[0])]
        train_y = np.array(result['target'])
        return train_x, train_y      # training set, training targets

    def predict_from_coefs(self, price_history, coef, intercept):
        """
        Predict next price by applying coefficients and intercept to price history.
        """
        if len(price_history) < len(coef):
            return -1

        pred = intercept
        for i, x in enumerate(price_history[-len(coef):]):
            pred += x["price"] * coef[i]
        return pred

    def get_own_trades_quant(self, state, product, price, is_buy=False, greater=False):
        quantity = 0
        if product not in state.own_trades:
            return quantity

        for trade in state.own_trades[product]:
            logger.print(f"price {trade.price} | quant {trade.quantity} | buyer {trade.buyer} | seller {trade.seller}")
            if is_buy and trade.buyer == "SUBMISSION":
                if greater and trade.price > price:
                    logger.print("trade price > given price")
                    quantity += trade.quantity
                elif not greater and trade.price < price:
                    logger.print("trade price < given price")
                    quantity += trade.quantity
            elif not is_buy and trade.seller == "SUBMISSION":
                if greater and trade.price < price:
                    logger.print("trade price > given price")
                    quantity += trade.quantity
                elif not greater and trade.price > price:
                    logger.print("trade price < given price")
                    quantity += trade.quantity

        logger.print(f"returned {quantity} quantity")
        return quantity

    def amethyst_algo(self, state, order_depth):
        orders: List[Order] = []

        buy_price = 9999
        sell_price = 10001

        curr_pos = state.position["AMETHYSTS"] if "AMETHYSTS" in state.position else 0
        ask_limit = self.POSITION_LIMITS["AMETHYSTS"] - curr_pos
        bid_limit = self.POSITION_LIMITS["AMETHYSTS"] + curr_pos

        highest_ask, _ = list(order_depth.sell_orders.items())[-1] if len(order_depth.sell_orders) != 0 else float('inf')
        lowest_bid, _ = list(order_depth.buy_orders.items())[-1] if len(order_depth.buy_orders) != 0 else 0

        # buying logic
        if len(order_depth.sell_orders) != 0:
            # market take
            for ask, amt in list(order_depth.sell_orders.items()):
                ask_amt = abs(amt)

                if ask_limit > 0 and int(ask) <= buy_price:
                    orders.append(Order("AMETHYSTS", ask, min(ask_amt, ask_limit)))
                    ask_limit -= min(ask_amt, ask_limit)
                elif ask_limit > 0 and curr_pos < 0 and int(ask) == buy_price + 1:
                    orders.append(Order("AMETHYSTS", ask, min(ask_amt, min(ask_limit, abs(curr_pos)))))
                    ask_limit -= min(ask_amt, min(ask_limit, abs(curr_pos)))

            # market make
            if ask_limit > 0:
                if curr_pos > 0:
                    orders.append(Order("AMETHYSTS", min(buy_price, lowest_bid + 1), ask_limit))
                else:
                    orders.append(Order("AMETHYSTS", min(buy_price, lowest_bid + 2), ask_limit))

        # selling logic
        if len(order_depth.buy_orders) != 0:
            # market take
            for bid, amt in list(order_depth.buy_orders.items()):
                bid_amt = abs(amt)

                if bid_limit > 0 and int(bid) >= sell_price:
                    orders.append(Order("AMETHYSTS", bid, -min(bid_amt, bid_limit)))
                    bid_limit -= min(bid_amt, bid_limit)
                elif bid_limit > 0 and curr_pos > 0 and int(bid) == sell_price - 1:
                    orders.append(Order("AMETHYSTS", bid, -min(bid_amt, min(bid_limit, abs(curr_pos)))))
                    bid_limit -= min(bid_amt, min(bid_limit, abs(curr_pos)))

            # market make
            if bid_limit > 0:
                if curr_pos < 0:
                    orders.append(Order("AMETHYSTS", max(sell_price, highest_ask - 1), -bid_limit))
                else:
                    orders.append(Order("AMETHYSTS", max(sell_price, highest_ask - 2), -bid_limit))

        return orders

    def starfruit_algo(self, state, order_depth, all_trade_history):
        orders: List[Order] = []

        # Values affecting linear regression to tune
        num_vars = 5
        default_coef = [0.21875239, 0.78025873]
        default_intercept = 5.003692924688039

        if "STARFRUIT" not in all_trade_history or len(all_trade_history["STARFRUIT"]) < len(default_coef):
            return orders

        # linear regression to predict next price
        if len(all_trade_history["STARFRUIT"]) >= num_vars * 2:
            train_x, train_y = self.preprocess_for_lr(all_trade_history["STARFRUIT"], num_vars)
            coefs, intercept = self.lin_regression(train_x, train_y)
            predicted_price = int(round(self.predict_from_coefs(all_trade_history["STARFRUIT"], coefs, intercept)))
        else:
            predicted_price = int(round(self.predict_from_coefs(all_trade_history["STARFRUIT"], default_coef, default_intercept)))

        buy_price = predicted_price - 1
        sell_price = predicted_price + 1
        logger.print(f"Starfruit predicted price is {predicted_price} | buy price is {buy_price} | sell price is {sell_price}")

        curr_pos = state.position["STARFRUIT"] if "STARFRUIT" in state.position else 0
        ask_limit = self.POSITION_LIMITS["STARFRUIT"] - curr_pos
        bid_limit = self.POSITION_LIMITS["STARFRUIT"] + curr_pos

        highest_ask, _ = list(order_depth.sell_orders.items())[-1] if len(order_depth.sell_orders) != 0 else float('inf')
        lowest_bid, _ = list(order_depth.buy_orders.items())[-1] if len(order_depth.buy_orders) != 0 else 0

        # buying logic
        if len(order_depth.sell_orders) != 0:
            # market take
            for ask, amt in list(order_depth.sell_orders.items()):
                ask_amt = abs(amt)

                if ask_limit > 0 and int(ask) <= buy_price:
                    orders.append(Order("STARFRUIT", ask, min(ask_amt, ask_limit)))
                    ask_limit -= min(ask_amt, ask_limit)
                elif ask_limit > 0 and curr_pos < 0 and int(ask) == predicted_price:
                    orders.append(Order("STARFRUIT", ask, min(ask_amt, min(ask_limit, abs(curr_pos)))))
                    ask_limit -= min(ask_amt, min(ask_limit, abs(curr_pos)))

            # market make
            if ask_limit > 0:
                orders.append(Order("STARFRUIT", min(buy_price, lowest_bid + 1), ask_limit))

        # selling logic
        if len(order_depth.buy_orders) != 0:
            # market take
            for bid, amt in list(order_depth.buy_orders.items()):
                bid_amt = abs(amt)

                if bid_limit > 0 and int(bid) >= sell_price:
                    orders.append(Order("STARFRUIT", bid, -min(bid_amt, bid_limit)))
                    bid_limit -= min(bid_amt, bid_limit)
                elif bid_limit > 0 and curr_pos > 0 and int(bid) == predicted_price:
                    orders.append(Order("STARFRUIT", bid, -min(bid_amt, min(bid_limit, abs(curr_pos)))))
                    bid_limit -= min(bid_amt, min(bid_limit, abs(curr_pos)))

            # market make
            if bid_limit > 0:
                orders.append(Order("STARFRUIT", max(sell_price, highest_ask - 1), -bid_limit))

        return orders

    def orchids_algo(self, state, order_depth):
        orders: List[Order] = []
        conversions = 0

        if "ORCHIDS" not in state.observations.conversionObservations:
            return orders

        observation = state.observations.conversionObservations["ORCHIDS"]
        foreign_bid = observation.bidPrice
        foreign_ask = observation.askPrice

        # calculate price we buy/sell at in order to be profitable if convert now
        profitable_bid = foreign_bid - observation.exportTariff - observation.transportFees
        profitable_ask = foreign_ask + observation.importTariff + observation.transportFees

        curr_pos = state.position["ORCHIDS"] if "ORCHIDS" in state.position else 0
        ask_limit = self.POSITION_LIMITS["ORCHIDS"]
        bid_limit = self.POSITION_LIMITS["ORCHIDS"]

        # buying logic
        orders.append(Order("ORCHIDS", min(math.floor(profitable_bid), math.floor(foreign_ask) + 1), ask_limit))

        # selling logic
        orders.append(Order("ORCHIDS", max(math.ceil(profitable_ask), math.ceil(foreign_bid) - 1), -bid_limit))

        # conversion logic
        conversions -= curr_pos

        logger.print(f"Profitable bid is {profitable_bid} | profitable ask is {profitable_ask} | requested conversion is {conversions}")
        return orders, conversions

    def run(self, state: TradingState):

        if state.traderData == "":
            price_history = {"AMETHYSTS": [], "STARFRUIT": []}
        else:
            price_history = json.loads(state.traderData)

        # update trade history
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]

            if product not in price_history:
                price_history[product] = []

            med_bid = np.median([p for p, v in list(order_depth.buy_orders.items()) for _ in range(abs(v))])
            med_ask = np.median([p for p, v in list(order_depth.sell_orders.items()) for _ in range(abs(v))])

            if len(order_depth.buy_orders) > 0 or len(order_depth.sell_orders) > 0:
                if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
                    mid_price = (med_bid + med_ask) / 2
                elif len(order_depth.buy_orders) > 0:
                    mid_price = med_bid
                else:
                    mid_price = med_ask

                price_history[product].append({
                    "timestamp": state.timestamp,
                    "price": mid_price,
                })

            # remove the oldest price history
            while len(price_history[product]) > self.MAX_HISTORY_LENGTH[product]:
                price_history[product].pop(0)

        # calculate orders for each product
        orders = {}
        conversions = 0
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            res: List[Order] = []
            conv = 0

            if product == "AMETHYSTS":
                res = self.amethyst_algo(state, order_depth)
            elif product == "STARFRUIT":
                res = self.starfruit_algo(state, order_depth, price_history)
            elif product == "ORCHIDS":
                res, conv = self.orchids_algo(state, order_depth)

            orders[product] = res
            conversions += conv

        trader_data = json.dumps(price_history)

        logger.flush(state, orders, conversions, "trader_data")
        return orders, conversions, trader_data
