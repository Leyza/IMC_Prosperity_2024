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

    def sma(self, price_history, history_length, curr_timestamp, pad_beginning=False, initial_avg=0):
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
    
    # exponential moving average
    def ema(self, price_history, history_length, curr_timestamp):
        total = 0
        count = 0

        if len(price_history) > 0 and price_history[0]["timestamp"] > curr_timestamp - history_length:
            padding_count = (price_history[0]["timestamp"] - curr_timestamp + history_length) / self.TIMESTAMP_INTERVAL
            for i in range(int(padding_count)):
                count += 1
                k = 2 / (count / self.TIMESTAMP_INTERVAL + 1)
                total = price_history[0]["price"] * k + total * (1 - k)

        for trade in price_history:
            if trade["timestamp"] < curr_timestamp - history_length:
                continue

            count += 1
            k = 2 / (count / self.TIMESTAMP_INTERVAL + 1)
            total = trade["price"] * k + total * (1 - k)

        return total

    def volatility(self, price_history, history_length, curr_timestamp, mean, pad_beginning=False, initial_sqr_residual=0.0):
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

    def lin_regression(self, price_history, history_length, curr_timestamp, pad_beginning=False):
        considered_trades = []
        if pad_beginning and price_history[0]["timestamp"] > curr_timestamp - history_length:
            considered_trades.extend([(i, price_history[0]["price"]) for i in range(curr_timestamp - history_length, price_history[0]["timestamp"], self.TIMESTAMP_INTERVAL)])

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

        mean = self.sma(price_history["AMETHYSTS"], 10000, state.timestamp, True, 10000)
        std = self.volatility(price_history["AMETHYSTS"], 8000, state.timestamp, mean, True, 1.2)
        logger.print(f"Amethyst mean is {mean} | std is {std}")

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
                    orders.append(Order("AMETHYSTS", math.floor(buy_price - std * 1.2), ask_limit))
                    break

        if len(order_depth.buy_orders) != 0:
            for bid, amt in list(order_depth.buy_orders.items()):
                bid_amt = abs(amt)

                if int(bid) >= sell_price and bid_limit > 0:
                    logger.print(f"AMETHYST SELL {str(min(bid_amt, bid_limit))}x, {bid}")
                    orders.append(Order("AMETHYSTS", bid, -min(bid_amt, bid_limit)))
                    bid_limit -= min(bid_amt, bid_limit)
                elif bid_limit > 0:
                    orders.append(Order("AMETHYSTS", math.ceil(sell_price + std * 1.2), -bid_limit))
                    break

        return orders

    def starfruit_algo(self, state, order_depth, all_trade_history):
        orders: List[Order] = []

        if "STARFRUIT" not in all_trade_history or len(all_trade_history["STARFRUIT"]) == 0:
            return orders

        mean = self.sma(all_trade_history["STARFRUIT"], 6000, state.timestamp, False, 5001)
        std = self.volatility(all_trade_history["STARFRUIT"], 6000, state.timestamp, mean, True, 3.5)
        logger.print(f"Starfruit mean is {mean} | std is {std}")

        m, b = self.lin_regression(all_trade_history["STARFRUIT"], 8000, state.timestamp)
        logger.print(f"Starfruit slope is {m}")

        predicted_price = m * (state.timestamp + self.TIMESTAMP_INTERVAL) + b
        logger.print(f"Starfruit predicted price is {predicted_price}")

        curr_pos = state.position["STARFRUIT"] if "STARFRUIT" in state.position else 0
        ask_limit = self.POSITION_LIMITS["STARFRUIT"] - curr_pos
        bid_limit = self.POSITION_LIMITS["STARFRUIT"] + curr_pos

        if len(order_depth.sell_orders) != 0:
            for ask, amt in list(order_depth.sell_orders.items()):
                ask_amt = abs(amt)

                if ask_limit > 0 and int(ask) < predicted_price - std * 0.2:
                    logger.print(f"STARFRUIT BUY {str(min(ask_amt, ask_limit))}x, {ask}")
                    orders.append(Order("STARFRUIT", ask, min(ask_amt, ask_limit)))
                    ask_limit -= min(ask_amt, ask_limit)
                elif ask_limit > 0:
                    orders.append(Order("STARFRUIT", math.floor(predicted_price - std), ask_limit))
                    break

        if len(order_depth.buy_orders) != 0:
            for bid, amt in list(order_depth.buy_orders.items()):
                bid_amt = abs(amt)

                if bid_limit > 0 and int(bid) > predicted_price + std * 0.2:
                    logger.print(f"STARFRUIT SELL {str(min(bid_amt, bid_limit))}x, {bid}")
                    orders.append(Order("STARFRUIT", bid, -min(bid_amt, bid_limit)))
                    bid_limit -= min(bid_amt, bid_limit)
                elif bid_limit > 0:
                    orders.append(Order("STARFRUIT", math.ceil(predicted_price + std), -bid_limit))
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