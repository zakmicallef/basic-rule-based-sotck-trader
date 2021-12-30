import pandas as pd
import talib
import numpy as np
import matplotlib.pyplot as plt

class Ledger:
    def __init__(self, cash):
        self.cash = cash
        self.ledger = pd.DataFrame([['hold', cash]], columns=['buy/sell/hold', 'cash'])

    def add(self, decision, cash):
        self.cash = cash
        self.ledger.loc[self.ledger.shape[0]] = [decision, self.cash]

    def add_pct(self, decision, pct):
        self.cash = self.cash * pct
        self.ledger.loc[self.ledger.shape[0]] = [decision, self.cash]

    def get_ledger(self):
        self.ledger['dayly return/losses percentage'] = self.ledger['cash'].pct_change().fillna(0) * 100
        self.ledger['dayly return/losses'] = self.ledger['cash'].diff().fillna(0)
        self.ledger['good trade'] = self.ledger['dayly return/losses percentage'] > 0
        return self.ledger


todo = [
    ...
]

df = pd.read_csv('dataset/AMZN.csv')
df = df[200:]


# Set number of days and standard deviations to use for rolling lookback period for Bollinger band calculation
def bollinger_strat(df, window, std):
    # Calculate rolling mean and standard deviation using number of days set above
    rolling_mean = df['Open'].rolling(window).mean()
    rolling_std = df['Open'].rolling(window).std()
    # create two new DataFrame columns to hold values of upper and lower Bollinger bands
    df['Rolling Mean'] = rolling_mean
    df['Bollinger High'] = rolling_mean + (rolling_std * std)
    df['Bollinger Low'] = rolling_mean - (rolling_std * std)

    df['Position'] = None
    df = df[-500:]
    # Fill our newly created position column - set to sell (-1) when the price hits the upper band, and set to buy (1) when it hits the lower band
    for row in range(len(df)):

        if (df['Open'].iloc[row] > df['Bollinger High'].iloc[row]) and (
                df['Open'].iloc[row - 1] < df['Bollinger High'].iloc[row - 1]):
            df['Position'].iloc[row] = -1

        if (df['Open'].iloc[row] < df['Bollinger Low'].iloc[row]) and (
                df['Open'].iloc[row - 1] > df['Bollinger Low'].iloc[row - 1]):
            df['Position'].iloc[row] = 1
        # Forward fill our position column to replace the "None" values with the correct long/short positions to represent the "holding" of our position

    # forward through time
    df['Position'].fillna(method='ffill', inplace=True)
    # Calculate the daily market return and multiply that by the position to determine strategy returns
    df['Market Return'] = np.log(df['Open'] / df['Open'].shift(1))
    df['Strategy Return'] = df['Market Return'] * df['Position']
    # Plot the strategy returns
    df['Strategy Return'].cumsum().plot(label='window ' + str(window) + ' std ' + str(std))
    return df['Strategy Return'].cumsum().iloc[-1]


ranges = []
for do in todo:
    df = pd.read_csv('dataset/' + do['fileName'])

    plt.figure(figsize=(12, 8))
    windows = [10, 20, 100]
    stds = [2, 3, 5]
    res = []
    for window in windows:
        for std in stds:
            res.append(bollinger_strat(df, window, std))
    plt.legend(loc='best')
    plt.title(do['stockName'])
    plt.savefig('resultsBB' + '/' + do['savePlace'] + '.png')
    plt.close('all')
    ranges.append((min(res), max(res)))

print(ranges)

# the RSI set at 7 day rolling window and the MACD Momentum strategy
def day_trader(df, title):
    df['rsi7'] = talib.RSI(df.Close.values, timeperiod=7)
    macd, macdsignal, df['macdhist'] = talib.MACD(df.Close.values, fastperiod=12, slowperiod=26, signalperiod=9)
    df = df[-70:]
    df['rsi7 over 50'] = df['rsi7'] > 50
    df['macdhist cross over'] = ((df['macdhist'] > 0).shift(-1)) & (df['macdhist'] < 0)
    df['macdhist cross to under'] = ((df['macdhist'] < 0)) & (df['macdhist'] > 0).shift(1)
    df['buy'] = df['rsi7 over 50'] & df['macdhist cross over']
    df['sell'] = (df['rsi7'] < 50) & df['macdhist cross to under']

    df['position'] = np.zeros(df.shape[0])
    position = 0
    for row in range(len(df)):
        if position == 2:
            position = 0

        if position == 0:
            if df['buy'].iloc[row]:
                position = 1
            elif df['sell'].iloc[row]:
                position = -1
        elif position == 1:
            if df['sell'].iloc[row]:
                position = -1
        elif position == -1:
            if df['sell'].iloc[row]:
                position = 1

        df['position'].iloc[row] = position

    df['action'] = df['position'].shift(1).fillna(0) != df['position']
    df['change'] = df['Close'] - df['Close'].shift(-1)
    df['pct prof/loss'] = np.zeros(df.shape[0])
    df['pct prof/loss'] = (df['change'] * df['position']) / df['Close']
    df['pct prof/loss'][df['action']] = 0

    leger_obj = Ledger(1000)

    des_pre = None
    for row in range(len(df)):
        pct = df['pct prof/loss'].iloc[row]
        des = df['position'].iloc[row]
        if des_pre is None:
            if des == 0:
                des_ = 'hold'
            elif des == 1:
                des_ = 'buy'
            elif des == -1:
                des_ = 'sell'
        else:
            if des == des_pre:
                des_ = 'hold'
            else:
                if des == 0:
                    des_ = 'hold'
                elif des == 1:
                    des_ = 'buy'
                elif des == -1:
                    des_ = 'sell'
        des_pre = des
        leger_obj.add_pct(des_, (pct + 1))

    leg = leger_obj.get_ledger()
    leg['cash'].plot(label=title)
    return leg


def sharpe(y):
    return np.sqrt(y.count()) * (y.mean() / y.std())

plt.figure(figsize=(24, 16))
for do in todo:
    df = pd.read_csv('dataset/' + do['fileName'])
    ledger = day_trader(df, do['stockName'])[:-1]

    description = pd.DataFrame([
        [
            'Technical Ind 1',
            ledger[ledger['buy/sell/hold'] == 'hold'].count()[0],
            ledger[ledger['buy/sell/hold'] == 'buy'].count()[0],
            ledger[ledger['buy/sell/hold'] == 'sell'].count()[0],
            ledger[(ledger['buy/sell/hold'] != 'hold') & (ledger['good trade'] == True)].count()[0],
            ledger[(ledger['buy/sell/hold'] != 'hold') & (ledger['good trade'] == False)].count()[0],
            ledger['cash'].iloc[-1],
            sharpe(ledger['dayly return/losses'])
        ]
    ], columns=['description', 'No trading days', 'Buy trades', 'Short trades', 'profit days', 'loss days',
                'End Amount (USD)', 'Sharp Radio'])

    ledger.to_csv('resultsRs7Macd/' + do['fileName'] + '.csv')
    description.to_csv('resultsRs7Macd/description_' + do['fileName'] + '.csv')

plt.title('compare RS7 with macdhist cross over')
plt.legend(loc="best")
plt.savefig('resultsRs7Macd/profit.png')
plt.close('all')
