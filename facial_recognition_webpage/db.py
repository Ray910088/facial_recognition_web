import sqlite3

def connect_to_db():
    # 連接到資料庫
    conn = sqlite3.connect(r'database.db')

    try:
        # 驗證連接是否成功
        cursor = conn.cursor()
        cursor.execute("SELECT SQLITE_VERSION()")

        # 提取並打印資料庫版本資訊
        version = cursor.fetchone()
        print(f"成功連接到 SQLite 資料庫，版本為: {version[0]}")

        return conn
    except sqlite3.Error as e:
        print(f"連接到 SQLite 資料庫出錯: {e}")


def get_user_full_name(user_id):
    conn = connect_to_db()
    cursor = conn.cursor()

    query = "SELECT full_name FROM users WHERE id = ?"
    cursor.execute(query, (user_id,))
    result = cursor.fetchone()

    conn.close()

    if result:
        return result[0]
    else:
        return "沒找到"