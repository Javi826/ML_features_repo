def SMA(df, col, n):
    df[f'SMA_{n}'] = df[col].rolling(window=n).mean()

    return df


# EMA
def EMA(df, col, n):
    df[f'EMA_{n}'] = df[col].ewm(span=n, adjust=False).mean()

    return df


# RSI
def RSI(df, col, n):
    rsi_series = ta.momentum.RSIIndicator(df[col], int(n)).rsi()
    df[f"RSI_{n}"] = rsi_series

    return df


# SMOOTH_RSI
def smooth_RSI(df, col, n, ma_type='SMA', ma_period=10):
    rsi_series = ta.momentum.RSIIndicator(df[col], int(n)).rsi()
    df[f"RSI_{n}"] = rsi_series
    if ma_type == 'SMA':
        df[f"RSI_{n}_{ma_type}_{ma_period}"] = df[f"RSI_{n}"].rolling(window=ma_period).mean()
    elif ma_type == 'EMA':
        df[f"RSI_{n}_{ma_type}_{ma_period}"] = df[f"RSI_{n}"].ewm(span=ma_period, adjust=False).mean()

    return df


# STO_RSI
def STO_RSI(df, col, n):
    StoRsi = ta.momentum.StochRSIIndicator(df[col], int(n))

    df[f"STO_RSI_{n}"] = StoRsi.stochrsi() * 100
    df[f"STO_RSI_D_{n}"] = StoRsi.stochrsi_d() * 100
    df[f"STO_RSI_K_{n}"] = StoRsi.stochrsi_k() * 100

    return df


# ATR
def ATR(df, n):
    df[f"ATR_{n}"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], n).average_true_range()

    return df


# SMOOTH ATR
def smooth_ATR(df, n, ma_type='SMA', ma_period=10):
    df[f"ATR_{n}"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], n).average_true_range()
    if ma_type == 'SMA':
        df[f"ATR_{n}_{ma_type}_{ma_period}"] = df[f"ATR_{n}"].rolling(window=ma_period).mean()
    elif ma_type == 'EMA':
        df[f"ATR_{n}_{ma_type}_{ma_period}"] = df[f"ATR_{n}"].ewm(span=ma_period, adjust=False).mean()

    return df


# MACD
def MACD(df, col, n_fast=12, n_slow=26, n_signal=9):
    # MACD
    macd_indicator = ta.trend.MACD(df[col], window_slow=n_slow, window_fast=n_fast, window_sign=n_signal)

    # MACD Columns
    df[f'MACD_{n_fast}_{n_slow}'] = macd_indicator.macd()
    df[f'MACD_signal_{n_signal}'] = macd_indicator.macd_signal()
    df[f'MACD_histogram_{n_fast}_{n_slow}_{n_signal}'] = macd_indicator.macd_diff()

    return df


# SMOOTH_MACD
def smooth_MACD(df, col, n_fast=12, n_slow=26, n_signal=9, ma_type='SMA', ma_period=10):
    macd_indicator = ta.trend.MACD(df[col], window_slow=n_slow, window_fast=n_fast, window_sign=n_signal)
    df[f'MACD_{n_fast}_{n_slow}'] = macd_indicator.macd()
    df[f'MACD_signal_{n_signal}'] = macd_indicator.macd_signal()
    df[f'MACD_histogram_{n_fast}_{n_slow}_{n_signal}'] = macd_indicator.macd_diff()

    # Smoothing
    if ma_type == 'SMA':
        df[f'MACD_histogram_{n_fast}_{n_slow}_{n_signal}_{ma_type}_{ma_period}'] = df[
            f'MACD_histogram_{n_fast}_{n_slow}_{n_signal}'].rolling(window=ma_period).mean()
    elif ma_type == 'EMA':
        df[f'MACD_histogram_{n_fast}_{n_slow}_{n_signal}_{ma_type}_{ma_period}'] = df[
            f'MACD_histogram_{n_fast}_{n_slow}_{n_signal}'].ewm(span=ma_period, adjust=False).mean()

    return df


# STK
def STK(df, close_col, low_col, high_col, n):
    stoch = ta.momentum.StochasticOscillator(high=df[high_col], low=df[low_col], close=df[close_col], window=n,
                                             smooth_window=10)

    df[f'STK_{n}'] = stoch.stoch()

    return df


# STD
def rolling_std(df, col, n):
    df[f'std_{n}'] = df[col].rolling(window=n).std()

    return df


# RSI_LOWESS
def rsi_lowess(df, col, n, frac=0.01):
    # RSI
    rsi_period = int(n)
    df['RSI'] = ta.momentum.RSIIndicator(df[col], window=rsi_period).rsi()

    df['RSI'].fillna(0, inplace=True)

    numerical_index = df.index.map(pd.Timestamp.timestamp)

    smoothed_rsi = lowess(df['RSI'], numerical_index, frac=frac, it=3)

    # RSI_Lowess
    df['RSI_lowess'] = smoothed_rsi[:, 1]

    return df


# CMMA
def CMMA(df, close_col, high_col, low_col, lookback, atr_length):
    n = len(df)
    output = np.zeros(n)
    front_bad = max(lookback, atr_length)

    for i in range(front_bad, n):
        # ATR
        tr = np.maximum(
            df[high_col].iloc[i - atr_length + 1:i + 1] - df[low_col].iloc[i - atr_length + 1:i + 1],
            np.abs(df[high_col].iloc[i - atr_length + 1:i + 1] - df[close_col].iloc[i - atr_length:i].shift(1)),
            np.abs(df[low_col].iloc[i - atr_length + 1:i + 1] - df[close_col].iloc[i - atr_length:i].shift(1))
        )
        atr = np.mean(tr)

        # LOG AVERAGE
        log_avg = np.mean(np.log(df[close_col].iloc[i - lookback:i]))

        if atr > 0:
            denom = atr * np.sqrt(lookback + 1)
            z_score = (np.log(df[close_col].iloc[i]) - log_avg) / denom
            output[i] = 100.0 * norm.cdf(z_score) - 50.0
        else:
            output[i] = 0.0

    df[f"CMMA_{lookback}"] = output

    return df


def kalman_filter(df, close_col, window=None):
    valores = df[close_col].values

    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=0,
                      initial_state_covariance=1,
                      observation_covariance=1,
                      transition_covariance=0.01)

    state_means, _ = kf.filter(valores)

    df[f"kalman_{close_col}"] = state_means

    if window:
        df[f"kalman_{close_col}_{window}"] = df[f"kalman_{close_col}"].rolling(window=window).mean()

    return df


# TREND
def calculate_trend(df, close_col, high_col, low_col, lookback, atr_length, trend_type):
    n = len(df)
    output = np.zeros(n)
    front_bad = max(lookback, atr_length)

    # Legendre simulation
    if trend_type == 'linear':
        coefs = np.linspace(-1, 1, lookback)
        col_name = 'linear_trend'
    elif trend_type == 'quadratic':
        coefs = np.linspace(-1, 1, lookback) ** 2
        col_name = 'quad_trend'
    elif trend_type == 'cubic':
        coefs = np.linspace(-1, 1, lookback) ** 3
        col_name = 'cubic_trend'
    else:
        raise ValueError("Tren no valid 'linear', 'quadratic' o 'cubic'.")
    # ATR (Average True Range)
    for icase in range(front_bad, n):

        tr = np.maximum(
            df[high_col].iloc[icase - atr_length + 1:icase + 1] - df[low_col].iloc[icase - atr_length + 1:icase + 1],
            np.abs(df[high_col].iloc[icase - atr_length + 1:icase + 1] - df[close_col].iloc[
                                                                         icase - atr_length:icase].shift(1)),
            np.abs(
                df[low_col].iloc[icase - atr_length + 1:icase + 1] - df[close_col].iloc[icase - atr_length:icase].shift(
                    1))
        )
        atr = np.mean(tr)

        # Log promedian
        log_prices = np.log(df[close_col].iloc[icase - lookback:icase])
        dot_prod = np.dot(log_prices, coefs)
        log_avg = np.mean(log_prices)

        if atr > 0:
            # Z-score based ATR
            denom = atr * np.sqrt(lookback + 1)
            z_score = (dot_prod - log_avg) / denom
            output[icase] = 100.0 * norm.cdf(z_score) - 50.0
        else:
            output[icase] = 0.0

    df[f"{col_name}_{lookback}"] = output
    return df


# ENTROPY
class Entropy:
    def __init__(self, wordlen):

        if wordlen < 1:
            self.ok = False
            self.bins = None
            return

        self.wordlen = wordlen
        self.nbins = 2 ** wordlen  # n bins ^wordlenght
        self.bins = np.zeros(self.nbins, dtype=int)  # bins initialilzed
        self.ok = True

    def entropy(self, nx, wordlen, x):

        if wordlen < 1 or nx <= wordlen:
            return 0.0

        # Bins
        self.bins.fill(0)

        # Count Bins
        for i in range(wordlen, nx):
            k = 1 if x[i - 1] > x[i] else 0
            for j in range(1, wordlen):
                k *= 2
                if x[i - j - 1] > x[i - j]:
                    k += 1
            self.bins[k] += 1

        # Entropy
        ent = 0.0
        total_bins = nx - wordlen
        for i in range(self.nbins):
            p = self.bins[i] / total_bins
            if p > 0.0:
                ent -= p * np.log(p)

        # Norm entropy
        return ent / np.log(self.nbins)


def entropy_filter(df, close_col, wordlen, window=None):
    valores = df[close_col].values

    # Init Entropy
    entropy_instance = Entropy(wordlen)

    # Entropy each point
    entropia = np.zeros(len(valores))
    for i in range(len(valores)):
        if i < wordlen:
            entropia[i] = 0  # First value no entropy
        else:
            entropia[i] = entropy_instance.entropy(i + 1, wordlen, valores[:i + 1][::-1])

    # Add column
    df[f"entropy_{close_col}"] = entropia

    # Promedian jic
    if window:
        df[f"entropy_{close_col}_{window}"] = df[f"entropy_{close_col}"].rolling(window=window).mean()

    return df


# PRICE INTENSITY
def price_intensity(df, col='close', n=10):

    if col not in df.columns:
        raise ValueError(f"La columna '{col}' no existe en el DataFrame.")

    close_col = df[col].values
    open_col = df['open'].values
    high_col = df['high'].values
    low_col = df['low'].values
    n_to_smooth = int(n + 0.5)

    if n_to_smooth < 1:
        n_to_smooth = 1

    n_data = len(df)
    output = np.zeros(n_data)

    # La primera barra no tiene barra previa
    denom = high_col[0] - low_col[0]
    if denom < 1.e-60:
        denom = 1.e-60
    output[0] = (close_col[0] - open_col[0]) / denom

    # Raw Values
    for icase in range(1, n_data):
        denom = high_col[icase] - low_col[icase]
        if high_col[icase] - close_col[icase - 1] > denom:
            denom = high_col[icase] - close_col[icase - 1]
        if close_col[icase - 1] - low_col[icase] > denom:
            denom = close_col[icase - 1] - low_col[icase]
        if denom < 1.e-60:
            denom = 1.e-60
        output[icase] = (close_col[icase] - open_col[icase]) / denom

    # Smoothing
    if n_to_smooth > 1:
        alpha = 2.0 / (n_to_smooth + 1.0)
        smoothed = output[0]
        for icase in range(1, n_data):
            smoothed = alpha * output[icase] + (1.0 - alpha) * smoothed
            output[icase] = smoothed

    # Compression
    for icase in range(n_data):
        output[icase] = 100.0 * norm.cdf(0.8 * np.sqrt(n_to_smooth) * output[icase]) - 50.0

    # Add column
    df[f"price_intensity_{n_to_smooth}"] = output
    return df


# PERCENTAGE PRICE OSCILLATOR (PPO)
def PPO(df, close_col, short_length, long_length, n_to_smooth=1):
    # Convert lengths to integers
    short_length = int(short_length + 0.5)
    long_length = int(long_length + 0.5)
    n_to_smooth = int(n_to_smooth + 0.5)

    # Initialize variables
    n = len(df)
    front_bad = long_length + n_to_smooth
    if front_bad > n:
        front_bad = n
    back_bad = 0

    long_alpha = 2.0 / (long_length + 1.0)
    short_alpha = 2.0 / (short_length + 1.0)

    long_sum = short_sum = df[close_col].iloc[0]
    output = np.zeros(n)
    output[0] = 0.0  # Poorly defined at first case

    # Compute PPO
    for icase in range(1, n):
        long_sum = long_alpha * df[close_col].iloc[icase] + (1.0 - long_alpha) * long_sum
        short_sum = short_alpha * df[close_col].iloc[icase] + (1.0 - short_alpha) * short_sum
        output[icase] = 100.0 * (short_sum - long_sum) / (long_sum + 1.e-15)  # Official PPO

        # Optional normalization (comment out if not needed)
        output[icase] = 100.0 * norm.cdf(0.2 * output[icase]) - 50.0

    # Smooth if requested
    if n_to_smooth > 1:
        alpha = 2.0 / (n_to_smooth + 1.0)
        smoothed = output[0]
        for icase in range(1, n):
            smoothed = alpha * output[icase] + (1.0 - alpha) * smoothed
            output[icase] -= smoothed

    # Store PPO result in DataFrame
    df[f"PPO_{short_length}_{long_length}"] = output

    return df


def create_features(df):
    df = SMA(df, col='close', n=10)
    df = EMA(df, col='close', n=10)
    df = RSI(df, col='close', n=10)
    df = price_intensity(df, col='close', n=10)
    df = price_intensity(df, col='close', n=15)
    df = price_intensity(df, col='close', n=20)
    df = STO_RSI(df, col='close', n=14)
    df = ATR(df, n=10)
    df = MACD(df, col='close', n_fast=12, n_slow=26, n_signal=9)
    df = rolling_std(df, col='close', n=10)
    df = STK(df, close_col='close', low_col='low', high_col='high', n=10)
    df = entropy_filter(df, 'close', wordlen=4, window=10)
    df = CMMA(df, 'close', 'high', 'low', lookback=10, atr_length=200)
    df = kalman_filter(df, 'close', window=10)
    df = calculate_trend(df, 'close', 'high', 'low', lookback=10, atr_length=200, trend_type='linear')
    df = calculate_trend(df, 'close', 'high', 'low', lookback=10, atr_length=200, trend_type='quadratic')
    df = calculate_trend(df, 'close', 'high', 'low', lookback=10, atr_length=200, trend_type='cubic')
    df = smooth_ATR(df, n=14, ma_type='SMA', ma_period=10)
    df = smooth_RSI(df, col='close', n=14, ma_type='EMA', ma_period=10)
    df = smooth_MACD(df, col='close', n_fast=12, n_slow=26, n_signal=9, ma_type='EMA', ma_period=10)

    return df