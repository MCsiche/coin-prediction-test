import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model, save_model
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.losses import Huber
import tensorflow as tf
import joblib

import cx_Oracle
from datetime import timedelta
import matplotlib.pyplot as plt

# 모델 경로
model_save_path = "D:/jms/final_project/models/BTCUSDT_model.h5"

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
        FROM TWENTY_DAYS_DATA R
        JOIN (SELECT CANDLE_DATE_TIME_KST, OPENING_PRICE, HIGH_PRICE, LOW_PRICE, TRADE_PRICE 
            FROM TWENTY_DAYS_DATA WHERE MARKET = 'KRW-USDT') U
            ON R.CANDLE_DATE_TIME_KST = U.CANDLE_DATE_TIME_KST
        WHERE R.MARKET = '{table_name}' AND ROWNUM <= (480*4)
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
    
def usdt_data_load():
    cur = conn.cursor()
    
    usdt_query = """
        SELECT 
            FIRST_VALUE(TRADE_PRICE) OVER() AS TRADE_PRICE
        FROM TWENTY_DAYS_DATA
        WHERE MARKET = 'KRW-USDT'
        """
        
    cur.execute(usdt_query)
    usdt_data = cur.fetchall()
    if not usdt_data:
            print("No data returned from SQL query.")
            return pd.DataFrame()  # 빈 DataFrame 반환
    df = pd.DataFrame(usdt_data, columns=['TRADE_PRICE'])
    cur.close()

    return df

# 테이블 생성 함수
# 테이블 초기화 함수
def initialize_oracle_table():
    cur = conn.cursor()
    
    create_query = """
            CREATE TABLE PREDICTION_RESULT (
                korean_name VARCHAR2(255),
                market VARCHAR2(255),
                candle_date_time_kst VARCHAR2(255),
                predict_price FLOAT

            )
        """
    
    cur.execute("SELECT COUNT(*) FROM USER_TABLES WHERE TABLE_NAME = 'PREDICTION_RESULT'")
    if cur.fetchone()[0] == 0:
        cur.execute(create_query)
        print("PREDICTION_RESULT 테이블 생성 완료")
        conn.commit()
    cur.execute("TRUNCATE TABLE PREDICTION_RESULT")
    cur.close()

initialize_oracle_table()
# 데이터 로드

features = ['open', 'high', 'low', 'close', 'volume', 'volume_close_ratio', 'volatility',
            'close_pct_change', 'volume_pct_change', 'high_low_ratio', 
            'moving_avg_10', 'hour_sin', 'hour_cos', 'week_sin', 'week_cos']


def add_derived_features(df):
    
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # df2['timestamp'] = pd.to_datetime(df2['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)  # 시간 순서로 정렬

    # 종가 변화율 (pct_change로 계산)
    df['close_pct_change'] = df['close'].pct_change().fillna(0)
    
    # 거래량 변화율
    df['volume_pct_change'] = df['volume'].pct_change().fillna(0)
    
    # 고가-저가 차이 비율
    df['high_low_ratio'] = (df['high'] - df['low']) / df['low']
    
    # 10개 이동 평균선
    df['moving_avg_10'] = df['close'].rolling(window=10).mean().fillna(df['close'])
    
    # 30개 이동 평균선
    df['moving_avg_30'] = df['close'].rolling(window=30).mean().fillna(df['close'])
    
    # 시간 관련 특성 추가
    df['hour'] = df.index.hour  # 시간
    df['day_of_week'] = df.index.dayofweek  # 주간 요일
    
    # 거래량 대비 종가 비율
    df['volume_close_ratio'] = df['volume'] / (df['close'] + 1e-6)  # NaN 방지
    
    # 20개 구간의 변동성 (표준편차)
    df['volatility'] = df['close'].rolling(window=20).std().fillna(0)
    
    # 시간의 주기적 특성 반영 (sin, cos)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # 주기적 변화를 반영 (0~6 범위를 0~2π로 변환 후 sin, cos 생성)
    df['week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    df = df[features]
    return df



# 실시간 데이터 예시
df1= load_real_time_data()  # 실시간 데이터 로드
if not df1.empty:
    df1 = add_derived_features(df1)  # 파생 변수 추가
df_usdt = usdt_data_load()



# 데이터 슬라이싱 (실시간 데이터)
three_minutes_per_day = 480  # 하루 3분 봉 개수
recent_days = 3  # 최근 180일
# df1 = df1[:three_minutes_per_day * recent_days]  # 최근 3일 데이터
# df1 = df1.iloc[::2, :]  # 6분 단위 샘플링

# # 최근 1주일 데이터를 df2로 분리
# recent_week_start = df1.index[-1] - pd.Timedelta(days=3)
# df2 = df1[df1.index >= recent_week_start]

# # df1에서 최근 1주일 데이터를 제외
# df1 = df1[df1.index < recent_week_start]
#-------------------------------------------------------------

# 데이터 스케일링
# 스케일러 로드
scaler = joblib.load("D:/jms/final_project/scalers/BTCUSDT_scaler.pkl")
scaled_data_1 = scaler.fit_transform(df1)

# 학습된 모델 로드
model = load_model(model_save_path)
print(scaler.scale_)
print(scaler.center_)
# 예측을 위한 데이터 준비
sequence_length = 480*3   # 1일치 데이터, 3분 간격
test_length = 60  # 3시간 후 예측
num_features = scaled_data_1.shape[1]

def predict_close(model, data, scaler, sequence_length, test_length):
    # 입력 데이터 준비 (마지막 sequence_length 길이 사용)
    X_test = np.array([data[-sequence_length:]])  # Shape: (1, sequence_length, num_features)

    # 모델 예측 수행
    predictions_scaled = model.predict(X_test).reshape(test_length, -1)  # Shape: (test_length, num_features)

    # 'close' 값만 추출 후 스케일 복원
    close_column_index = features.index('close')  # 'close'의 인덱스 확인
    predicted_close_scaled = predictions_scaled[:, close_column_index]  # 'close' 값만 선택
    # 스케일 복원 (RobustScaler의 경우)
    # RobustScaler 복원 공식: 원본 값 = (스케일 값 * scale_) + center_
    
    dummy_data = np.zeros((predicted_close_scaled.shape[0], len(features)))  # 더미 배열 생성
    dummy_data[:, close_column_index] = predicted_close_scaled  # 'close' 값만 삽입

    # 스케일 복원
    predicted_close = scaler.inverse_transform(dummy_data)[:, close_column_index]

    return predicted_close


# 예측 실행
predicted_close = predict_close(model, scaled_data_1, scaler, sequence_length, test_length)



# usdt -> krw
predicted_close = predicted_close*df_usdt.iloc[0].values

# x축: 예측 시간 (예: 3시간 동안의 시간 간격)
# 마지막 데이터의 타임스탬프
last_timestamp = df1.index[-1]  # 최근 데이터의 마지막 시간

# 3분 간격 타임스탬프 생성
xticks = [last_timestamp + timedelta(minutes=3 * i) for i in range(test_length)]


# 타임스탬프를 'YYYY-MM-DDThh:mm:ss' 형태로 변환
xticks_str = [ts.strftime('%Y-%m-%dT%H:%M:%S') for ts in xticks]
print(xticks_str)  # 생성된 타임스탬프 확인

def save_prediction_to_db(xticks, predicted_prices, table_name="KRW-BTC"):
    try:
        cur = conn.cursor()
        
        conn.commit()
        # 여러 데이터를 한 번에 삽입할 때 사용할 쿼리
        insert_query = """
            INSERT INTO PREDICTION_RESULT (korean_name, market, candle_date_time_kst, predict_price)
            SELECT cn.korean_name, :1, :2, :3
            FROM coin_name cn
            WHERE cn.market = :1
        """

        # 하나의 execute로 여러 데이터를 삽입하기 위해 파라미터 리스트를 준비
        insert_data = [(table_name, timestamp, price) for timestamp, price in zip(xticks, predicted_prices)]

        # execute_batch를 사용하여 여러 데이터를 한 번에 삽입
        
        cx_Oracle.Cursor.executemany(cur, insert_query, insert_data)
        conn.commit()
        print("예측 결과가 Oracle DB에 저장되었습니다.")
        cur.close()
    except cx_Oracle.DatabaseError as e:
        print("DB 저장 중 오류 발생:", e)

# 마지막 실측 close 값 (스케일링 전 값)
last_close_value = df1['close'].iloc[-1]

# # 예측된 스케일링된 close 값을 조정
# predicted_close_adjusted = predicted_close + last_close_value

# db 저장 함수 실행
save_prediction_to_db(xticks_str, predicted_close)

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(xticks, predicted_close, label='Predicted Close', color='b', marker='o')

# 그래프 제목 및 레이블 설정
plt.title("Predicted Close Prices (Next 3 Hours)", fontsize=16)
plt.xlabel("Time Steps (3-minute intervals)", fontsize=12)
plt.ylabel("Predicted Close Price", fontsize=12)

# 범례 표시
plt.legend()
plt.show()

