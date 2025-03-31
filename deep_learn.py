import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.losses import Huber
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf

# 데이터 불러오기
file_path_1 = "D:/jms/final_project/year_data/binance_data_BTCUSDT.csv"
# file_path_2 = "D:/jms/final_project/year_data/binance_data_BTCUSDT_lastweek.csv"

df1 = pd.read_csv(file_path_1)
# df2 = pd.read_csv(file_path_2)

# 주요 컬럼 정리
df1 = df1[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
# df2 = df2[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

# 시간 정렬 및 정리
df1['timestamp'] = pd.to_datetime(df1['timestamp'])
# df2['timestamp'] = pd.to_datetime(df2['timestamp'])
df1.set_index('timestamp', inplace=True)
# df2.set_index('timestamp', inplace=True)

# ---------------------------------------------------------------------------
# 파생 변수 추가
df1['close_pct_change'] = df1['close'].pct_change().fillna(0)  # 종가 변화율
df1['volume_pct_change'] = df1['volume'].pct_change().fillna(0)  # 거래량 변화율
df1['high_low_ratio'] = (df1['high'] - df1['low']) / df1['low']  # 고가와 저가 차이 비율
df1['moving_avg_10'] = df1['close'].rolling(window=10).mean().fillna(df1['close'])  # 10개 이동 평균선
df1['moving_avg_30'] = df1['close'].rolling(window=30).mean().fillna(df1['close'])  # 30개 이동 평균선

# 시간 관련 파생 변수 추가
df1['hour'] = df1.index.hour
# df2['hour'] = df2.index.hour

df1['day_of_week'] = df1.index.dayofweek  # 주말 여부 (0: 평일, 1: 주말)

# 새로운 변수 추가
df1['volume_close_ratio'] = df1['volume'] / (df1['close'] + 1e-6)  # NaN 방지

# 20개 구간의 변동성(표준편차) 계산
df1['volatility'] = df1['close'].rolling(window=20).std().fillna(0)  # NaN 제거


# 시간의 주기적 특성 반영
df1['hour_sin'] = np.sin(2 * np.pi * df1['hour'] / 24)
df1['hour_cos'] = np.cos(2 * np.pi * df1['hour'] / 24)


# 주기적 변화를 반영 (0~6 범위를 0~2π로 변환 후 sin, cos 생성)
df1['week_sin'] = np.sin(2 * np.pi * df1['day_of_week'] / 7)
df1['week_cos'] = np.cos(2 * np.pi * df1['day_of_week'] / 7)




# 필요한 컬럼만 선택
features = ['open', 'high', 'low', 'close', 'volume', 'volume_close_ratio', 'volatility',
            'close_pct_change', 'volume_pct_change', 'high_low_ratio', 
            'moving_avg_10', 'hour_sin', 'hour_cos', 'week_sin', 'week_cos']

df1 = df1[features]
# 데이터 슬라이싱
three_minutes_per_day = 480
recent_days = 30
df1 = df1[-three_minutes_per_day * recent_days:]
# df1 = df1.iloc[::2, :]  # 6분 단위 샘플링
# 최근 1주일 데이터를 df2로 분리
recent_week_start = df1.index[-1] - pd.Timedelta(days=3)
df2 = df1[df1.index >= recent_week_start]
# df1에서 최근 1주일 데이터를 제외
df1 = df1[df1.index < recent_week_start]
# 데이터 스케일링
# df1과 df2를 합쳐 fit 후 transform
scaler = RobustScaler()
# fit은 두 데이터셋을 합친 데이터에 대해 수행
combined_data = pd.concat([df1, df2], axis=0)
# 스케일러를 두 데이터셋을 합친 데이터에 대해 fit
scaler.fit(combined_data)
# transform은 각 데이터셋에 별도로 적용
scaled_data_1 = scaler.transform(df1)
scaled_data_2 = scaler.transform(df2)
# df1의 마지막 행과 df2의 첫 번째 행의 차이를 계산
last_value_df1 = scaled_data_1[-1]
first_value_df2 = scaled_data_2[0]

# 차이를 보정하여 연속성 유지
adjustment = last_value_df1 - first_value_df2
scaled_data_2 += adjustment

# 데이터 생성 함수
def create_train_val_sequences(data, sequence_length, test_length):
    X_train, y_train, X_test, y_test = [], [], [], []
    step = 10
    for i in range(0, len(data) - sequence_length - test_length, step):
        train_seq = data[i:i + sequence_length]
        test_seq = data[i + sequence_length:i + sequence_length + test_length]
        
        if len(test_seq) == test_length:
            X_train.append(train_seq)
            y_train.append(train_seq)  # open, high, low, close
            X_test.append(test_seq)
            y_test.append(test_seq)  # open, high, low, close
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)



# 데이터 생성
sequence_length = three_minutes_per_day*3  # 3일
test_length = 60  # 3시간 (60개)
num_features = scaled_data_1.shape[1]
X_train, y_train, X_test, y_test = create_train_val_sequences(scaled_data_1, sequence_length, test_length)
# y_train의 차원 수정: (batch_size, test_length * num_features)
y_train = y_train.reshape(-1, test_length * num_features)


# 모델 생성
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.4),
    LSTM(128, return_sequences=False),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dense(test_length * num_features)  # 출력 차원을 test_length * num_features로 설정
])

# 손실 함수 및 옵티마이저 설정
huber = Huber(delta=1.0)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps= 480 * 7,
    decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss=huber)

# 조기종료 및 가중치 decay_factor 적용
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
decay_factor=0.9
weights = np.exp(-np.arange(len(y_train)) / decay_factor)

# 학습
model.fit(
    X_train, y_train, 
    epochs=50, 
    batch_size=32, 
    validation_split=0.25,
    callbacks=[early_stopping],
    verbose=1,
    sample_weight=weights
)
# 예측 후 모델에서 예측값 가져오기
test_predictions = model.predict(X_test)

# 예측값을 (batch_size, test_length, num_features) 형태로 변환
test_predictions = test_predictions.reshape(-1, 60,15)

# 'close' 값만 추출 (예: 'close'가 5번째 변수라면, 해당 인덱스를 선택)
close_column_index = features.index('close')  # 'close'가 있는 인덱스를 추출
test_close_predictions = test_predictions[:, :, close_column_index]  # 'close' 값만 선택

    # 학습 후 모델 저장
model_save_path = "D:/jms/final_project/models/BTCUSDT_model.h5"
scaler_save_path = "D:/jms/final_project/scalers/BTCUSDT_scaler.pkl"

# 학습된 모델 저장 (전체 저장)
model.save(model_save_path)
# 스케일러 저장

joblib.dump(scaler, scaler_save_path)


# 두 번째 데이터셋 처리

# df2의 마지막 시간
last_timestamp = df2.index[-1]  # df2의 마지막 timestamp

# 두 번째 데이터셋 처리
if len(scaled_data_2) >= sequence_length:
    final_week_data = scaled_data_2[-sequence_length:]
    current_input = final_week_data

    # 3시간 예측
    next_hours_predictions = []
    
    current_input_reshaped = np.expand_dims(current_input, axis=0)
    
    # 예측된 값을 model을 통해 얻음
    predicted_values = model.predict(current_input_reshaped).flatten()
    
    next_hours_predictions.append(predicted_values)
    
    # 슬라이딩 윈도우 방식에서 예측값을 새로운 데이터에 추가하지 않음
    # 대신, 현재 시점의 데이터만을 사용하여 예측을 반복함
    current_input = current_input[1:]  # current_input에서 가장 오래된 데이터를 제거
    # 예측값을 새 데이터에 포함시키지 않고, 기존의 current_input을 유지

    # 예측된 값을 다음 시간 예측값 배열로 변환 (test_length, 15개의 변수)
    next_hours_predictions_array = np.array(next_hours_predictions)
    # next_hours_predictions_array의 shape을 (60, 15)로 변경
    next_hours_predictions_array = next_hours_predictions_array.reshape(-1, num_features)
    # 예상되는 형태는 (60, 15) 즉, 60개 시간 동안의 예측 결과가 15개의 특성으로 되어야 합니다.
    # scaler의 inverse_transform에 맞는 형태로 데이터를 변환
    next_hours_predictions_restored = scaler.inverse_transform(next_hours_predictions_array)

    # 'close' 값만 추출
    next_hours_predictions_restored_close = next_hours_predictions_restored[:, features.index('close')]

    # 시각화: 예측된 'close' 값
    plt.figure(figsize=(12, 6))
    x_ticks = [last_timestamp + pd.Timedelta(minutes=3 * i) for i in range(len(next_hours_predictions_restored_close))]
    plt.scatter(x_ticks, next_hours_predictions_restored_close, color='red', label='Predicted Close Prices', s=10)  # 점그래프 (size=s=10)
    plt.title("Next 3 Hours Predicted Close Prices")
    plt.xlabel("Time")
    plt.ylabel("Close Price")
    plt.xticks(ticks=x_ticks[::6], labels=[time.strftime('%Y-%m-%d %H:%M:%S') for time in x_ticks[::6]], rotation=45)  # 6번째마다 시간 표시
    plt.legend()
    plt.grid()
    plt.show()

else:
    print("Error: Not enough data in the second dataset for prediction.")