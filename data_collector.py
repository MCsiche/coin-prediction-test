import pandas as pd
import requests
import json
import time
import threading
from tkinter import Tk, Label, Button, Text, Scrollbar
from datetime import datetime, timedelta
import pybithumb



# 마켓 리스트
markets = [
    "BTC", "ETH", "XRP", "DOGE", "ADA",
    "SAND", "SHIB", "GLM", "SOL", "XLM",
    "BICO", "ELF", "WLD", "PEPE", "COS",
    "ENS", "ETC", "SEI", "BNB", "TRX", "USDT"
]

# Excel 파일 저장 경로
def get_excel_path():
    today_date = datetime.now().strftime('%Y-%m-%d')
    return f"real_time_data_{today_date}.xlsx"

# Oracle DB 설정
import cx_Oracle
try:
    dsn = cx_Oracle.makedsn(host="192.168.60.2", port=1521, sid="xe")
    conn = cx_Oracle.connect(user="BDV", password="bdv0328", dsn=dsn)
    print("Oracle DB 연결 성공")
except cx_Oracle.Error as e:
    print("Oracle DB 연결 실패", e)


# 테이블 초기화 함수
def initialize_oracle_table():
    cur = conn.cursor()
    create_queries = {
        "K_REAL_TIME": """
            CREATE TABLE K_REAL_TIME (
                korean_name VARCHAR2(255),
                market VARCHAR2(255),
                candle_date_time_utc VARCHAR2(255),
                candle_date_time_kst VARCHAR2(255),
                opening_price FLOAT,
                high_price FLOAT,
                low_price FLOAT,
                trade_price FLOAT,
                timestamp NUMBER,
                candle_acc_trade_price FLOAT,
                candle_acc_trade_volume FLOAT,
                unit VARCHAR2(50)

            )
        """,
        "REAL_TIME_TEMP": """
            CREATE TABLE REAL_TIME_TEMP (
                market VARCHAR2(255),
                candle_date_time_utc VARCHAR2(255),
                candle_date_time_kst VARCHAR2(255),
                opening_price FLOAT,
                high_price FLOAT,
                low_price FLOAT,
                trade_price FLOAT,
                timestamp NUMBER,
                candle_acc_trade_price FLOAT,
                candle_acc_trade_volume FLOAT,
                unit VARCHAR2(50)
            )
        """,
        "TWENTY_DAYS_DATA": """
            CREATE TABLE TWENTY_DAYS_DATA (
                korean_name VARCHAR2(255),
                market VARCHAR2(255),
                candle_date_time_utc VARCHAR2(255),
                candle_date_time_kst VARCHAR2(255),
                opening_price FLOAT,
                high_price FLOAT,
                low_price FLOAT,
                trade_price FLOAT,
                timestamp NUMBER,
                candle_acc_trade_price FLOAT,
                candle_acc_trade_volume FLOAT,
                unit VARCHAR2(50)
            )
        """
    }
        
    for table_name, create_query in create_queries.items():
        cur.execute(f"SELECT COUNT(*) FROM USER_TABLES WHERE TABLE_NAME = '{table_name}'")
        if cur.fetchone()[0] == 0:
            cur.execute(create_query)
            print(f"{table_name} 테이블 생성 완료.")
        conn.commit()
    cur.close()

# 실시간 데이터를 받아와 Excel 저장
def fetch_and_save_data(retry_markets=None):
    cur = conn.cursor()

    # REAL_TIME_TEMP 삽입 쿼리
    insert_temp_query = """
                INSERT INTO REAL_TIME_TEMP (
                    market, candle_date_time_utc, candle_date_time_kst, opening_price,
                    high_price, low_price, trade_price, timestamp,
                    candle_acc_trade_price, candle_acc_trade_volume, unit
                ) VALUES (
                    :1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11
                )
                """
    # REAL_TIME_TEMP에서 K_REAL_TIME로 데이터 이동
    insert_join_query = """
                INSERT INTO K_REAL_TIME
                SELECT c.korean_name, t.market, t.candle_date_time_utc, t.candle_date_time_kst,
                    t.opening_price, t.high_price, t.low_price, t.trade_price, t.timestamp,
                    t.candle_acc_trade_price, t.candle_acc_trade_volume, t.unit
                FROM coin_name c
                JOIN REAL_TIME_TEMP t
                ON c.market = t.market
                """
    # TWENTY_DAYS_DATA로 데이터 추가
    insert_twenty_days_query = """
                INSERT INTO TWENTY_DAYS_DATA
                SELECT c.korean_name, t.market, t.candle_date_time_utc, t.candle_date_time_kst,
                    t.opening_price, t.high_price, t.low_price, t.trade_price, t.timestamp,
                    t.candle_acc_trade_price, t.candle_acc_trade_volume, t.unit
                FROM coin_name c
                JOIN REAL_TIME_TEMP t
                ON c.market = t.market
                """
    # 21일보다 오래된 TWENTY_DAYS_DATA 내의 DATA 삭제
    delete_old_query = """
                DELETE FROM TWENTY_DAYS_DATA
                WHERE timestamp < (
                    SELECT MIN(timestamp)
                    FROM (
                        SELECT timestamp
                        FROM (
                            SELECT timestamp
                            FROM TWENTY_DAYS_DATA
                            ORDER BY timestamp DESC
                        )
                        WHERE ROWNUM <= 10080
                    )
                )
                """

    excel_path = get_excel_path()
    if retry_markets is None:
        retry_markets = {}

    try:
        existing_df = pd.read_excel(excel_path)
    except FileNotFoundError:
        existing_df = pd.DataFrame()
    # TEMP 테이블 데이터 삭제
    cur.execute("TRUNCATE TABLE REAL_TIME_TEMP")
    conn.commit()
    all_data=[]
    for market in markets:
        try:
            # 데이터 가져오기
                data = pybithumb.get_ohlcv(market, payment_currency="KRW", interval="minute3")
                if data is None or data.empty:
                    print(f"{market} 데이터가 비어 있음. 건너뜁니다.")
                    continue
                
                # 마지막 1개 데이터 추출
                df = data.tail(1)
                df.loc[:,'market'] = f"KRW-{market}"
                df.reset_index(inplace=True)
                
                df.loc[:, 'market'] = f"KRW-{market}"
                df.loc[:, 'candle_date_time_utc'] = (pd.to_datetime(df['time'], unit='s') - pd.Timedelta(hours=9)).dt.strftime('%Y-%m-%dT%H:%M:%S')
                df.loc[:, 'candle_date_time_kst'] = pd.to_datetime(df['time'], unit='s').dt.strftime('%Y-%m-%dT%H:%M:%S')
                df['timestamp'] = pd.to_datetime(df['candle_date_time_kst']).astype(int) // 10**6
                
                # 필요한 데이터 처리
                df['candle_acc_trade_price'] = df['close'] * df['volume']
                df['candle_acc_trade_volume'] = df['volume']
                df['unit'] = 3  # 3분 단위로 설정

                # 컬럼 순서와 이름 맞추기
                df = df[['market', 'candle_date_time_utc', 'candle_date_time_kst',
                        'open', 'high', 'low', 'close', 'timestamp',
                        'candle_acc_trade_price', 'candle_acc_trade_volume', 'unit']]
                df.columns = ['market', 'candle_date_time_utc', 'candle_date_time_kst',
                            'opening_price', 'high_price', 'low_price', 'trade_price',
                            'timestamp', 'candle_acc_trade_price', 'candle_acc_trade_volume', 'unit']
                # DB에 데이터 삽입
                
                
                # REAL_TIME_TEMP 추가
                
                data_tuples = [tuple(row) for row in df.to_numpy()]
                cur.executemany(insert_temp_query, data_tuples)
                conn.commit()

                

                

                # Excel 추가
                existing_df = pd.concat([existing_df, df], ignore_index=True)
                existing_df.to_excel(excel_path, index=False)

                # 실패한 마켓 관리
                retry_markets.pop(market, None)

        except requests.exceptions.RequestException as e:
            log_message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 오류 발생: {str(e)}"
            update_log_gui(log_message)

            retry_markets[market] = retry_markets.get(market, 0) + 1
            if retry_markets[market] > 15:
                log_message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {market} 15회 재시도 실패, 중단"
                update_log_gui(log_message)
                retry_markets.pop(market, None)

    # # K_REAL_TIME 추가
    cur.execute(insert_join_query)
    conn.commit()

    # # 21일치 데이터 추가
    cur.execute(insert_twenty_days_query)
    conn.commit()

    # # 21일 초과 데이터 삭제
                
    cur.execute(delete_old_query)
    conn.commit()

    log_message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 데이터 수집 완료"
    update_log_gui(log_message)
    cur.close()
    return retry_markets
    
# GUI 로그 업데이트
def update_log_gui(message):
    log_text.insert("1.0", message + "\n")
    log_text.yview("end")
    lines = log_text.get("1.0", "end-1c").split('\n')
    if len(lines) > 5:  # 로그를 최대 5줄로 제한
        log_text.delete("5.0", "end")


# 데이터 수집 스레드
def start_data_collection():
    retry_markets = {}
    next_run_time = datetime.now().replace(second=0, microsecond=0)

    # 첫 실행 시간: 현재 시간 기준 다음 3분 간격으로 설정
    if next_run_time.minute % 3 != 1:
        next_run_time += timedelta(minutes=(3 - (next_run_time.minute % 3) + 1))  # '1분, 4분, 7분...'

    while not stop_event.is_set():
        current_time = datetime.now()

        # 현재 시간이 다음 실행 시간에 도달했는지 확인
        if current_time >= next_run_time:
            print(f"데이터 수집 시작: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            retry_markets = fetch_and_save_data(retry_markets)

            # 다음 실행 시간을 3분 후로 설정
            next_run_time += timedelta(minutes=3)

        # 남은 시간 계산 및 대기
        time_to_sleep = (next_run_time - datetime.now()).total_seconds()

        # 남은 대기 시간(0초 이상)만큼 대기
        if time_to_sleep > 0:
            print(f"다음 실행까지 대기: {time_to_sleep:.2f}초")
            time.sleep(time_to_sleep)
        else:
            # 예상보다 실행이 오래 걸렸을 때, 바로 다음 주기로 패스
            print("시간이 지체되어 다음 주기 바로 실행")
            next_run_time = datetime.now().replace(second=0, microsecond=0) + timedelta(minutes=3)





# GUI 종료 함수
def stop_program():
    stop_event.set()
    root.quit()

# GUI 생성
root = Tk()
root.title("Real-Time Data Collector")
root.geometry("500x400")

Label(root, text="실시간 데이터 수집", font=("Arial", 16)).pack(pady=10)

log_text = Text(root, height=15, width=60, font=("Arial", 12))
log_text.pack(pady=10)
scrollbar = Scrollbar(root, command=log_text.yview)
scrollbar.pack(side="right", fill="y")
log_text.config(yscrollcommand=scrollbar.set)

Button(root, text="종료", font=("Arial", 14), command=stop_program).pack(pady=10)

# 데이터 수집 시작
stop_event = threading.Event()
initialize_oracle_table()  # Oracle 테이블 초기화 (주석 처리)
thread = threading.Thread(target=start_data_collection)
thread.daemon = True
thread.start()

root.mainloop()
