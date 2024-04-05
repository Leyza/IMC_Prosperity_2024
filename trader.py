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
    POSITION_LIMITS = {"AMETHYSTS": 20, "STARFRUIT": 20}
    MAX_HISTORY_LENGTH = 30000
    TIMESTAMP_INTERVAL = 100

    def mean_price(self, price_history, initial_mean=0, initial_weight=1):
        total = initial_mean * initial_weight
        count = initial_weight
        for price in price_history:
            total += price["price"]
            count += 1

        return total / count

    def stddev_price(self, price_history, mean, initial_pop_size=1):
        total = 0
        count = initial_pop_size

        for price in price_history:
            total += (price["price"] - mean) ** 2
            count += 1

        return np.sqrt(total / count)

    def moving_average(self, price_history, history_length, curr_timestamp, initial_avg=0, initial_weight=1):
        total = initial_avg * initial_weight
        count = initial_weight

        for trade in price_history:
            if trade["timestamp"] < curr_timestamp - history_length:
                total = 0
                count = 0  # have enough data points to disregard initial
                continue

            total += trade["price"]
            count += 1

        return total / count

    def moving_stddev(self, price_history, history_length, curr_timestamp, mean, initial_pop_size=1):
        total = 0
        count = initial_pop_size

        for trade in price_history:
            if trade["timestamp"] < curr_timestamp - history_length:
                count = 0  # have enough data points to disregard initial
                continue

            total += (trade["price"] - mean) ** 2
            count += 1

        return np.sqrt(total / count)

    def lin_regression(self, price_history, history_length, curr_timestamp):
        considered_trades = []
        for trade in price_history:
            if trade["timestamp"] < curr_timestamp - history_length:
                continue

            considered_trades.append((trade["timestamp"], trade["price"]))

        x = [k for (k, v) in considered_trades]
        y = [v for (k, v) in considered_trades]
        A = np.vstack([x, np.ones(len(x))]).T

        m, b = np.linalg.lstsq(A, y, rcond=None)[0]

        return m, b  # slope, intercept

    def amethyst_algo(self, state, order_depth, price_history):
        orders: List[Order] = []

        if "AMETHYSTS" not in price_history or len(price_history["AMETHYSTS"]) == 0:
            return orders

        mean = self.moving_average(price_history["AMETHYSTS"], 20000, state.timestamp, 10000, 2000)
        std = self.moving_stddev(price_history["AMETHYSTS"], 20000, state.timestamp, mean, 2000)
        logger.print(f"mean is {mean} | std is {std}")

        buy_price = mean - std
        sell_price = mean + std

        curr_pos = state.position["AMETHYSTS"] if "AMETHYSTS" in state.position else 0
        ask_limit = self.POSITION_LIMITS["AMETHYSTS"] - curr_pos
        bid_limit = self.POSITION_LIMITS["AMETHYSTS"] + curr_pos

        if len(order_depth.sell_orders) != 0:
            for ask, amt in list(order_depth.sell_orders.items()):
                ask_amt = abs(amt)

                if int(ask) <= buy_price and ask_limit > 0:
                    logger.print(f"AMETHYST BUY {str(min(ask_amt, ask_limit))}x, {ask}")
                    orders.append(Order("AMETHYSTS", ask, min(ask_amt, ask_limit)))
                    ask_limit -= min(ask_amt, ask_limit)
                elif ask_limit > 0:
                    orders.append(Order("AMETHYSTS", math.floor(buy_price), ask_limit))
                    break

        if len(order_depth.buy_orders) != 0:
            for bid, amt in list(order_depth.buy_orders.items()):
                bid_amt = abs(amt)

                if int(bid) >= sell_price and bid_limit > 0:
                    logger.print(f"AMETHYST SELL {str(min(bid_amt, bid_limit))}x, {bid}")
                    orders.append(Order("AMETHYSTS", bid, -min(bid_amt, bid_limit)))
                    bid_limit -= min(bid_amt, bid_limit)
                elif bid_limit > 0:
                    orders.append(Order("AMETHYSTS", math.ceil(sell_price), -bid_limit))
                    break

        return orders

    def starfruit_algo(self, state, order_depth, all_trade_history):
        orders: List[Order] = []

        if "STARFRUIT" not in all_trade_history or len(all_trade_history["STARFRUIT"]) == 0 or \
                all_trade_history["STARFRUIT"][0]["timestamp"] > state.timestamp - 5000:
            return orders

        m, b = self.lin_regression(all_trade_history["STARFRUIT"], 8000, state.timestamp)
        # m1, b1 = self.lin_regression(all_trade_history["STARFRUIT"], 30000, state.timestamp)
        logger.print(f"slope1 for starfruit is {m} at time {state.timestamp}")
        # logger.print(f"slope2 for starfruit is {m1} at time {state.timestamp}")

        predicted_price = m * (state.timestamp + self.TIMESTAMP_INTERVAL) + b
        # overall_predicted_price = m1 * (state.timestamp + self.TIMESTAMP_INTERVAL) + b1

        curr_pos = state.position["STARFRUIT"] if "STARFRUIT" in state.position else 0
        ask_limit = self.POSITION_LIMITS["STARFRUIT"] - curr_pos
        bid_limit = self.POSITION_LIMITS["STARFRUIT"] + curr_pos

        if len(order_depth.sell_orders) != 0:
            for ask, amt in list(order_depth.sell_orders.items()):
                ask_amt = abs(amt)

                if ask_limit > 0 and m > 0 and int(ask) < predicted_price:
                    logger.print(f"STARFRUIT BUY {str(min(ask_amt, ask_limit))}x, {ask}")
                    orders.append(Order("STARFRUIT", ask, min(ask_amt, ask_limit)))
                    ask_limit -= min(ask_amt, ask_limit)
                elif ask_limit > 0:
                    orders.append(Order("STARFRUIT", math.floor(predicted_price) - 1, ask_limit))
                    break

        if len(order_depth.buy_orders) != 0:
            for bid, amt in list(order_depth.buy_orders.items()):
                bid_amt = abs(amt)

                if bid_limit > 0 and m < 0 and int(bid) > predicted_price:
                    logger.print(f"STARFRUIT SELL {str(min(bid_amt, bid_limit))}x, {bid}")
                    orders.append(Order("STARFRUIT", bid, -min(bid_amt, bid_limit)))
                    bid_limit -= min(bid_amt, bid_limit)
                elif bid_limit > 0:
                    orders.append(Order("STARFRUIT", math.ceil(predicted_price) + 1, -bid_limit))
                    break

        return orders

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

            if len(order_depth.buy_orders) > 0 or len(order_depth.sell_orders) > 0:
                if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
                    mid_price = np.average([int(list(order_depth.buy_orders.items())[0][0]),
                                            int(list(order_depth.sell_orders.items())[0][0])])
                elif len(order_depth.buy_orders) > 0:
                    mid_price = list(order_depth.buy_orders.items())[0]
                else:
                    mid_price = list(order_depth.sell_orders.items())[0]

                price_history[product].append({
                    "timestamp": state.timestamp,
                    "price": mid_price,
                })

            # remove oldest price history
            if len(price_history[product]) > 0:
                earliest_trade = price_history[product][0]
                while earliest_trade["timestamp"] < state.timestamp - self.MAX_HISTORY_LENGTH:
                    price_history[product].pop(0)

                    if len(price_history[product]) == 0:
                        break
                    earliest_trade = price_history[product][0]

        # calculate orders for each product
        orders = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            res: List[Order] = []

            if product == "AMETHYSTS":
                res = self.amethyst_algo(state, order_depth, price_history)
            elif product == "STARFRUIT":
                res = self.starfruit_algo(state, order_depth, price_history)

            orders[product] = res

        trader_data = json.dumps(price_history)
        conversions = 0

        logger.flush(state, orders, conversions, "trader_data")
        return orders, conversions, trader_data