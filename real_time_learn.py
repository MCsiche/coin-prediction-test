import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler
import time
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
import tensorflow as tf
import cx_Oracle
from tensorflow.keras.losses import Huber

# 모델과 스케일러 경로
model_save_path = "D:/jms/final_project/models/BTCUSDT_model.h5"
scaler_save_path = "D:/jms/final_project/scalers/BTCUSDT_scaler.pkl"
checkpoint_path = "D:/jms/final_project/models/BTCUSDT_model_checkpoint.h5"

# 모델 로드
model = load_model(model_save_path)

# 스케일러 로드
scaler = joblib.load(scaler_save_path)

#df1 로드

# Oracle DB 설정
try:
    dsn = cx_Oracle.makedsn(host="192.168.60.19", port=1521, sid="xe")
    conn = cx_Oracle.connect(user="BDV", password="bdv0328", dsn=dsn)
    print("Oracle DB 연결 성공")
except cx_Oracle.Error as e:
    print("Oracle DB 연결 실패", e)

# 실시간 데이터 로드 함수
def load_real_time_data():
    try:
        cur = conn.cursor()
        table_name = 'KRW-BTC'
        query = f"""
        SELECT 
            R.CANDLE_DATE_TIME_KST AS TIMESTAMP, 
            R.OPENING_PRICE / U.OPENING_PRICE AS OPEN, 
            R.HIGH_PRICE / U.HIGH_PRICE AS HIGH, 
            R.LOW_PRICE / U.LOW_PRICE AS LOW, 
            R.trade_price / U.TRADE_PRICE AS CLOSE, 
            R.CANDLE_ACC_TRADE_VOLUME AS VOLUME  
        FROM K_REAL_TIME R
        JOIN (SELECT CANDLE_DATE_TIME_KST, OPENING_PRICE, HIGH_PRICE, LOW_PRICE, TRADE_PRICE 
            FROM K_REAL_TIME WHERE MARKET = 'KRW-USDT') U
            ON R.CANDLE_DATE_TIME_KST = U.CANDLE_DATE_TIME_KST
        WHERE R.MARKET = '{table_name}' AND ROWNUM <= (1501)
        ORDER BY TIMESTAMP DESC
        """
        
        cur.execute(query)
        data = cur.fetchall()

        
        if not data:
            print("No data returned from SQL query.")
            return pd.DataFrame()  # 빈 DataFrame 반환
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        cur.close()
        return df
    except cx_Oracle.DatabaseError as e:
        print("Database error:", e)
        return pd.DataFrame()  # 빈 DataFrame 반환

# 학습에 사용할 features
features = ['open', 'high', 'low', 'close', 'volume', 'volume_close_ratio', 'volatility',
            'close_pct_change', 'volume_pct_change', 'high_low_ratio', 
            'moving_avg_10', 'hour_sin', 'hour_cos', 'week_sin', 'week_cos']

# 반복작업을 위한 함수
def predict_and_train_loop():
    while True:
        # 데이터 로드 (여기서 df1은 오라클 DB에서 불러오는 데이터를 의미)
        # df1 = load_your_data_from_oracle() # 필요에 맞게 데이터를 불러오세요.
        df1 = load_real_time_data()
        
        # df1에서 필요한 파생 변수를 생성

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

        # 필요한 칼럼만 선택
        df1 = df1[features]

        

        # 예측 및 훈련에 사용할 데이터 생성
        
        
        three_minutes_per_day = 480
        recent_days = 3
        # 예시: 예측할 기간을 60으로 설정
        sequence_length = three_minutes_per_day*3 # 3일 (시간 단위로 나눈 후 필요한 길이로 조정)
        test_length = 60  # 예측 3시간 (60개)
        num_features = 15

        df1 = df1[-three_minutes_per_day * recent_days-test_length:]
        print(df1.shape)
        #scaler
        scaler.fit(df1)
        scaled_data_1 = scaler.transform(df1)
        
        
        ##
        
        X_train, y_train, X_test, y_test = create_train_val_sequences(
            scaled_data_1, 
            sequence_length, 
            test_length,
            num_features)
        X_train = X_train.reshape(X_train.shape[0], -1)
        y_train = y_train.reshape(y_train.shape[0], -1)
        print("Adjusted X_train shape:", X_train.shape)
        print("Adjusted y_train shape:", y_train.shape)
        print("Adjusted X_train shape:", X_test.shape)
        print("Adjusted y_train shape:", y_test.shape)
        
        # 학습
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        decay_factor=0.9
        weights = np.exp(-np.arange(len(y_train)) / decay_factor)

        # 모델 훈련 (예측 후 학습)
        model.fit(
            X_train, y_train, 
            epochs=10, 
            batch_size=1, 
            validation_data=(X_test, y_test), 
            callbacks = [early_stopping],
            verbose=1,
            sample_weight = weights)

        # 예측
        # test_predictions = model.predict(X_test)
        
        # # 예측된 값 출력 (예: 'close' 값만 추출)
        # close_column_index = features.index('close')
        # test_close_predictions = test_predictions[:, :, close_column_index]  # 'close' 값만 선택
        


        # 1시간 대기
        time.sleep(3600)  # 1시간 대기

        # 모델 저장
        model.save(model_save_path)  # 체크포인트 모델 저장
    # 데이터 생성 함수
def create_train_val_sequences(data, sequence_length, test_length, num_features):
    # 데이터가 numpy 배열이 아닐 경우 변환
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    # 데이터가 1D 배열일 경우 reshape
    if len(data.shape) == 1:
        data = data.reshape(-1, num_features)  # 1D 데이터면 feature 개수에 맞게 reshape
    
    # 데이터의 길이가 sequence_length + test_length 이상이어야 합니다.
    if len(data) < sequence_length + test_length:
        raise ValueError(f"데이터 길이가 sequence_length + test_length보다 짧습니다. 데이터 길이: {len(data)}, 필요 길이: {sequence_length + test_length}")

    # 훈련 데이터와 테스트 데이터를 나눕니다.
    train_seq = data[-(sequence_length + test_length):-test_length, :]
    test_seq = data[-test_length:, :]
    
    # 데이터 차원 확장 (배치 차원 추가)
    X_train = np.expand_dims(train_seq, axis=0)  # (1, sequence_length, num_features)
    y_train = np.expand_dims(train_seq, axis=0)  # (1, sequence_length, num_features)
    X_test = np.expand_dims(test_seq, axis=0)    # (1, test_length, num_features)
    y_test = np.expand_dims(test_seq, axis=0)    # (1, test_length, num_features)
    
    return X_train, y_train, X_test, y_test



# 학습/예측 반복 시작
predict_and_train_loop()
