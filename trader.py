from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any
import numpy as np
import pandas as pd
import jsonpickle as js
import json
import math
from math import erf, sqrt, log, exp



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
    POSITION_LIMITS = {"AMETHYSTS": 20, "STARFRUIT": 20, "ORCHIDS": 100, "CHOCOLATE": 250, "STRAWBERRIES": 350, "ROSES": 60, "GIFT_BASKET": 60, "COCONUT": 300, "COCONUT_COUPON": 600}
    MAX_HISTORY_LENGTH = {"AMETHYSTS": 0, "STARFRUIT": 50, "ORCHIDS": 0,  "CHOCOLATE": 0, "STRAWBERRIES": 0, "ROSES": 0, "GIFT_BASKET": 101, "COCONUT": 100, "COCONUT_COUPON": 101}
    TIMESTAMP_INTERVAL = 100

    def sma(self, price_history, length):
        """
        Calculate the simple moving average on the price history.
        """
        prices = pd.Series(price_history[-length:])

        return prices.mean()
    
    def ema(self, price_history, length):
        """
        Calculate the exponential moving average on the price history.
        """
        total = 0
        count = 0

        for price in price_history[-length:]:
            count += 1
            k = 2 / (count + 1)
            total = price * k + total * (1 - k)

        return total

    def macd(self, price_history, short_length, long_length, sma=True):
        if sma:
            short = self.sma(price_history, short_length)
            long = self.sma(price_history, long_length)
        else:
            short = self.ema(price_history, short_length)
            long = self.ema(price_history, long_length)

        return short - long

    def price_volatility(self, price_history, length):
        """
        Calculate the historical price volatility.
        """
        prices = pd.Series(price_history[-length:])

        return prices.std()

    def returns_volatility(self, price_history, length):
        """
        Calculate the historical returns volatility.
        """
        prices = pd.Series(price_history[-length:])
        returns = np.log(prices[1:] / prices[:-1])

        return np.std(returns)

    def roc(self, price_history, distance=1):
        """
        Price rate of change in %.
        """
        if len(price_history) < distance + 1:
            return 0

        return (price_history[-1] - price_history[-1 - distance]) / price_history[-1 - distance] * 100

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
            past_vals.append(price_history[i])

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
            pred += x * coef[i]
        return pred

    def phi(self, x):
        """Cumulative distribution function for the standard normal distribution"""
        return (1.0 + erf(x / sqrt(2.0))) / 2.0

    def black_scholes(self, price, K, std, r, dt):
        d_plus = (np.log(price / K) + dt * (r + std ** 2 / 2)) / (sqrt(dt) * std)
        d_minus = d_plus - sqrt(dt) * std
        call_price = self.phi(d_plus) * price - self.phi(d_minus) * K * exp(-r * dt)

        return call_price

    def delta(self, price, K, std, r, dt):
        d_plus = (np.log(price / K) + dt * (r + std ** 2 / 2)) / (sqrt(dt) * std)
        return self.phi(d_plus)

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
            predicted_price = (list(order_depth.sell_orders.items())[-1][0] + list(order_depth.buy_orders.items())[-1][0]) / 2
        # linear regression to predict next price
        elif len(all_trade_history["STARFRUIT"]) >= num_vars * 2:
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

    def gift_basket_algo(self, state, order_depth, price_history):
        orders: List[Order] = []

        open_spread = 90
        close_spread = -5

        choco_orders = state.order_depths["CHOCOLATE"]
        straw_orders = state.order_depths["STRAWBERRIES"]
        rose_orders = state.order_depths["ROSES"]

        choco_price = (list(choco_orders.buy_orders.items())[0][0] + list(choco_orders.sell_orders.items())[0][0]) / 2
        straw_price = (list(straw_orders.buy_orders.items())[0][0] + list(straw_orders.sell_orders.items())[0][0]) / 2
        rose_price = (list(rose_orders.buy_orders.items())[0][0] + list(rose_orders.sell_orders.items())[0][0]) / 2

        combined_price = 4 * choco_price + 6 * straw_price + rose_price + 360
        gift_price = (list(order_depth.buy_orders.items())[0][0] + list(order_depth.sell_orders.items())[0][0]) / 2

        curr_pos = state.position["GIFT_BASKET"] if "GIFT_BASKET" in state.position else 0
        ask_limit = self.POSITION_LIMITS["GIFT_BASKET"] - curr_pos
        bid_limit = self.POSITION_LIMITS["GIFT_BASKET"] + curr_pos

        best_ask = list(order_depth.sell_orders.items())[0][0] if len(order_depth.sell_orders) > 0 else float('inf')
        best_bid = list(order_depth.buy_orders.items())[0][0] if len(order_depth.buy_orders) > 0 else 0

        # buying logic
        if ask_limit > 0:
            # close short positions
            if curr_pos < 0:
                orders.append(Order("GIFT_BASKET", min(math.ceil(combined_price) + close_spread, best_ask), min(abs(curr_pos), ask_limit)))
                ask_limit -= min(abs(curr_pos), ask_limit)

            orders.append(Order("GIFT_BASKET", min(math.floor(combined_price) - open_spread, best_ask), ask_limit))

        # selling logic
        if bid_limit > 0:
            # close long positions
            if curr_pos > 0:
                orders.append(Order("GIFT_BASKET", max(math.floor(combined_price) - close_spread, best_bid), -min(abs(curr_pos), bid_limit)))
                bid_limit -= min(abs(curr_pos), bid_limit)

            orders.append(Order("GIFT_BASKET", max(math.ceil(combined_price) + open_spread, best_bid), -bid_limit))

        logger.print(f"Gift basket combined price {combined_price} | current price {gift_price} | open spread: {open_spread} | close spread: {close_spread}")
        return orders

    def rose_algo(self, state, order_depth):
        orders: List[Order] = []

        curr_pos = state.position["ROSES"] if "ROSES" in state.position else 0
        ask_limit = self.POSITION_LIMITS["ROSES"] - curr_pos
        bid_limit = self.POSITION_LIMITS["ROSES"] + curr_pos

        best_ask = list(order_depth.sell_orders.items())[0][0] if len(order_depth.sell_orders) > 0 else float('inf')
        best_bid = list(order_depth.buy_orders.items())[0][0] if len(order_depth.buy_orders) > 0 else 0

        trades = state.market_trades["ROSES"] if "ROSES" in state.market_trades else []
        for trade in trades:
            if trade.buyer == "Rhianna" and trade.seller != "Rhianna":
                orders.append(Order("ROSES", best_ask, ask_limit))
            if trade.seller == "Rhianna" and trade.buyer != "Rhianna":
                orders.append(Order("ROSES", best_bid, -bid_limit))

        return orders

    def coconut_coupon_algo(self, state, order_depth, price_history):
        orders: List[Order] = []

        if "COCONUT" not in price_history or len(price_history["COCONUT"]) == 0:
            return orders

        pred_price = 0.5026 * price_history['COCONUT'][-1] - 4390.57

        curr_pos = state.position["COCONUT_COUPON"] if "COCONUT_COUPON" in state.position else 0
        ask_limit = self.POSITION_LIMITS["COCONUT_COUPON"] - curr_pos
        bid_limit = self.POSITION_LIMITS["COCONUT_COUPON"] + curr_pos

        best_ask = list(order_depth.sell_orders.items())[0][0] if len(order_depth.sell_orders) > 0 else float('inf')
        best_bid = list(order_depth.buy_orders.items())[0][0] if len(order_depth.buy_orders) > 0 else 0

        # buying logic
        orders.append(Order("COCONUT_COUPON", min(math.floor(pred_price) - 1, best_bid + 1), ask_limit))

        # selling logic
        orders.append(Order("COCONUT_COUPON", max(math.ceil(pred_price) + 1, best_ask - 1), -bid_limit))

        return orders

    def run(self, state: TradingState):
        data = json.loads(state.traderData) if state.traderData != "" else {"price_history": {}, "positions": {}}

        # update price history
        price_history = data["price_history"]
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

                price_history[product].append(mid_price)

            # remove the oldest price history
            while len(price_history[product]) > self.MAX_HISTORY_LENGTH[product]:
                price_history[product].pop(0)

        # update positions
        positions = data["positions"]
        for product, trades in list(state.market_trades.items()) + list(state.own_trades.items()):
            if product not in positions:
                positions[product] = {}

            for trade in trades:
                if trade.timestamp != state.timestamp - self.TIMESTAMP_INTERVAL:
                    continue

                if trade.buyer not in positions[product]:
                    positions[product][trade.buyer] = 0
                if trade.seller not in positions[product]:
                    positions[product][trade.seller] = 0

                positions[product][trade.buyer] += trade.quantity
                positions[product][trade.seller] -= trade.quantity

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
            elif product == "GIFT_BASKET":
                res = self.gift_basket_algo(state, order_depth, price_history)
            elif product == "ROSES":
                res = self.rose_algo(state, order_depth)
            elif product == "COCONUT_COUPON":
                res = self.coconut_coupon_algo(state, order_depth, price_history)

            orders[product] = res
            conversions += conv

        # serialize trader data
        data["price_history"] = price_history
        data["positions"] = positions
        trader_data = json.dumps(data)

        logger.flush(state, orders, conversions, "trader_data")
        return orders, conversions, trader_data
