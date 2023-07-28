import sqlite3

from PIL import ImageDraw, ImageFont, Image
from flask import Flask, render_template, request, redirect, url_for, Response, session, flash
import cv2
import numpy as np
import os
import dlib
from datetime import datetime
from db import connect_to_db, get_user_full_name, get_subject_name, get_subject_teacher
from face_recognition import face_recognition_eye_blink, face_recognition_eye_blink_img

app = Flask(__name__)
app.secret_key = 'yu_jay_is_handsome'

camera = None

def open_camera():
    global camera
    camera = cv2.VideoCapture(0)
    camera.set(3, 640)  # 設定攝像頭寬度
    camera.set(4, 480)  # 設定攝像頭高度
    return camera

def capture_by_frames(camera):
    detector = cv2.CascadeClassifier('model_landmarks/haarcascade_frontalface_default.xml')

    while True:
        success, frame = camera.read()  # read the camera frame

        frame = cv2.flip(frame, 1)  # 設定影像左右互換
        faces = detector.detectMultiScale(frame, 1.2, 6)

        # 繪製每張臉周圍的矩形
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue  # 跳過圖像編碼錯誤

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    if camera is not None:
        camera.release()  # 釋放鏡頭資源

    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        id = request.form['id']
        password = request.form['password']
        identity = request.form['identity']

        conn = connect_to_db()
        cursor = conn.cursor()

        # 在 users table 中查詢帳號和密碼的組合
        query = "SELECT * FROM users WHERE id = ? AND password = ? AND identity = ?"
        cursor.execute(query, (id, password, identity))
        result = cursor.fetchone()

        if result:
            # 驗證成功，設置 session
            session['user_id'] = result[1]
            session['user_identity'] = result[9]
            print(result[9])
            return redirect('/home')
        else:
            # 驗證失敗，返回登入頁面並顯示錯誤訊息
            error_message = '帳號/密碼/身分錯誤'
            return render_template('login.html', error_message=error_message)

    return render_template('login.html')

@app.route('/home')
def home():
    user_id = session.get('user_id')
    user_identity = session.get('user_identity')
    full_name = get_user_full_name(user_id)
    if user_identity == 'student':
        subject_names = get_subject_name(user_id)
        print(subject_names)

        # 取得當前日期時間
        current_day = datetime.now().strftime('%w')
        current_time = datetime.now().strftime('%H:%M')
        print(current_day)
        print(current_time)

        conn = connect_to_db()
        cursor = conn.cursor()

        for subject in subject_names:
            # 資料庫中有欄位名稱為start和finish，分別儲存課程開始和結束時間
            query = "SELECT * FROM subject WHERE day = ? AND start <= ? AND finish >= ? AND name = ?"
            cursor.execute(query, (current_day, current_time, current_time, subject))
            result = cursor.fetchone()
            if result is not None:
                break

        if result:
            # 顯示當下有課的課程資訊，再根據需要做相應的處理
            print(f"你目前有課程 '{subject}' 在 {result[5]} 開始，即將前往該課程並執行臉部辨識簽到！")
            session['subjectMain'] = subject
        else:
            print("你目前沒課喔！")

        conn.close()

    elif user_identity == 'teacher':
        subject_names = get_subject_teacher(user_id)
        print(subject_names)

        # 取得當前日期時間
        current_day = datetime.now().strftime('%w')
        current_time = datetime.now().strftime('%H:%M')
        print(current_day)
        print(current_time)

        conn = connect_to_db()
        cursor = conn.cursor()

        for subject in subject_names:
            # 資料庫中有欄位名稱為start和finish，分別儲存課程開始和結束時間
            query = "SELECT * FROM subject WHERE day = ? AND start <= ? AND finish >= ? AND name = ?"
            cursor.execute(query, (current_day, current_time, current_time, subject))
            result = cursor.fetchone()
            if result is not None:
                break

        if result:
            # 顯示當下有課的課程資訊，再根據需要做相應的處理
            print(f"您目前有課程 '{subject}' 在 {result[5]} 開始，即將前往該課程頁面！")
            session['subjectMain'] = subject
        else:
            print("您目前沒課喔！")

        conn.close()

    return render_template('home.html', full_name=full_name, identity=user_identity, subject_names=subject_names, result=result)

@app.route('/courseList')
def courseList():
    if request.referrer == "http://127.0.0.1:5000/record":
        referrer = session.get('referrer')
    else:
        referrer = request.referrer
        session['referrer'] = referrer
    print(referrer)

    user_id = session.get('user_id')
    user_identity = session.get('user_identity')
    full_name = get_user_full_name(user_id)
    if user_identity == 'student':
        subject_names = get_subject_name(user_id)
        print(subject_names)

    elif user_identity == 'teacher':
        subject_names = get_subject_teacher(user_id)
        print(subject_names)

    return render_template('courseList.html', full_name=full_name, identity=user_identity, subject_names=subject_names, referrer=referrer)

@app.route('/classes', methods=['GET', 'POST'])
def classes():
    user_id = session.get('user_id')
    user_identity = session.get('user_identity')
    full_name = get_user_full_name(user_id)
    if session.get('subjectMain'):
        subject = session.get('subjectMain')

    if request.method == 'POST':
        if 'subjectMain' in request.form:
            subject = request.form['subjectMain']
            session['subjectMain'] = subject
        else:
            subject = session.get('subjectMain')
            recognition_name = face_recognition_eye_blink(camera)
            print(full_name, recognition_name)
            if full_name == recognition_name:
                flash('簽到成功', 'success')

                # 連接到資料庫
                conn = connect_to_db()
                cursor = conn.cursor()

                # 獲取當前日期和時間
                now = datetime.now()
                date_time = now.strftime('%Y-%m-%d %H:%M:%S')

                # 插入到 roll_call_record 表中
                query = "INSERT INTO roll_call_record (stu_id, sub_name, date_time) VALUES (?, ?, ?)"
                cursor.execute(query, (user_id, subject, date_time,))
                conn.commit()  # 提交資料
                # 插入到 temp_record 表中
                query2 = "INSERT INTO temp_record (stu_id, sub_name, date_time) VALUES (?, ?, ?)"
                cursor.execute(query2, (user_id, subject, date_time,))
                conn.commit()  # 提交資料

                # 關閉資料庫連接
                if conn:
                    conn.close()

            else:
                flash('簽到失敗', 'error')

    return render_template('classes.html', full_name=full_name, identity=user_identity, subject=subject)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        department = request.form['department']
        grade = request.form['grade']
        id = request.form['id']
        password = request.form['password']
        gender = request.form['gender']
        full_name = request.form['full_name']
        email = request.form['email']
        phone = request.form['phone']
        address = request.form['address']
        print(department, grade, id, password, gender, full_name, email, phone, address)

        # 連接到資料庫
        conn = connect_to_db()

        try:
            # 創建游標
            cursor = conn.cursor()

            # 新增資料
            cursor.execute(
                "INSERT INTO users (department, id, password, grade, sex, full_name, email, phone, address, identity, su) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'student', 'F')",
                (department, id, password, grade, gender, full_name, email, phone, address))

            # 提交變更
            conn.commit()

            print("成功新增資料到 users 資料表")

        except sqlite3.Error as e:
            print(f"新增資料到 users 資料表時發生錯誤: {e}")

        finally:
            # 關閉資料庫連接
            if conn:
                conn.close()

        return redirect(url_for('detect_face', id=id))

    return render_template('register.html')

@app.route('/detect_face')
def detect_face():
    detector = cv2.CascadeClassifier('model_landmarks/haarcascade_frontalface_default.xml')
    count = 0
    id = request.args.get('id')

    while (True):
        success, frame = camera.read()  # read the camera frame

        frame = cv2.flip(frame, 1)  # 設定影像左右互換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 轉換成灰階
        faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)  # 辨識影像

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 加上紅框
            count += 1
            cv2.imwrite("images/User." + str(id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])  # 儲存影像到dataset資料夾

        k = cv2.waitKey(100) & 0xff  # 等待0.1秒，偵測鍵盤按鍵是否按下
        if k == 27:  # 按下ESC按鍵，中斷while迴圈
            break
        elif count >= 30:  # 偵測30張臉後，中斷while迴圈
            break
    print("\n 偵測完成")
    if camera is not None:
        camera.release()  # 釋放鏡頭資源

    return render_template('result.html', id=id)

@app.route('/train_model')
def train_model_route():
    path = 'images'
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    def getFaceAndID(path):
        detector = cv2.CascadeClassifier("model_landmarks/haarcascade_frontalface_default.xml")
        images = [os.path.join(path, f) for f in os.listdir(path)]
        FaceList = []
        IDList = []
        for image in images:
            img = Image.open(image).convert('L')  # 轉換成灰階
            img_np = np.array(img, 'uint8')
            id = int(os.path.split(image)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_np)
            for (x, y, w, h) in faces:
                FaceList.append(img_np[y:y + h, x:x + w])
                IDList.append(id)
        return FaceList, IDList

    print("\n影像辨識中")
    face, id = getFaceAndID(path)
    recognizer.train(face, np.array(id))
    recognizer.write('train/train.yml')  # 儲存訓練結果
    print("\n訓練出{0}張臉".format(len(np.unique(id))))
    return redirect('/')

@app.route('/record', methods=['GET', 'POST'])
def record():
    if request.method == 'POST':
        if 'subject' in request.form:
            subject = request.form['subject']
            session['subject'] = subject

    # 獲取當前使用者的 ID 和科目
    user_id = session.get('user_id')
    subject = session.get('subject')
    user_identity = session.get('user_identity')
    full_name = get_user_full_name(user_id)

    # 連接到資料庫
    conn = connect_to_db()
    cursor = conn.cursor()

    if user_identity == "student":
        # 查詢 roll_call_record 表中特定學生和該科目的簽到記錄
        query = "SELECT date_time FROM roll_call_record WHERE stu_id = ? AND sub_name = ?"
        cursor.execute(query, (user_id, subject))
        records = cursor.fetchall()

    elif user_identity == "teacher":
        # 查詢 roll_call_record 表中所有學生和該科目的簽到記錄
        query = """
                SELECT roll_call_record.stu_id, users.full_name, roll_call_record.date_time
                FROM roll_call_record
                INNER JOIN users ON roll_call_record.stu_id = users.id
                WHERE roll_call_record.sub_name = ?
            """
        cursor.execute(query, (subject,))
        records = cursor.fetchall()

    conn.close()

    return render_template('record.html',full_name=full_name, subject=subject, identity=user_identity, records=records)

@app.route('/video_capture')
def video_capture():
    return Response(capture_by_frames(open_camera()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/face_result')
def face_result():
    return Response(face_recognition_eye_blink_img(open_camera()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

if __name__ == '__main__':
    app.run()