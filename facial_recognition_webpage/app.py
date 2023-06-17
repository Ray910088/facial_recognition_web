import sqlite3

from PIL import ImageDraw, ImageFont, Image
from flask import Flask, render_template, request, redirect, url_for, Response, session, flash
import cv2
import numpy as np
import os
import dlib
from db import connect_to_db, get_user_full_name
from face_recognition import face_recognition_eye_blink, face_recognition_eye_blink_img

app = Flask(__name__)
camera = None
app.secret_key = 'yu_jay_is_handsome'

def capture_by_frames():
    global camera
    camera = cv2.VideoCapture(0)
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
    global camera
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
            session['user_identity'] = result[8]
            return redirect('/home')
        else:
            # 驗證失敗，返回登入頁面並顯示錯誤訊息
            error_message = '帳號/密碼/身分錯誤'
            return render_template('login.html', error_message=error_message)

    return render_template('login.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
    user_id = session.get('user_id')
    full_name = get_user_full_name(user_id)

    if request.method == 'POST':
        recognition_name = face_recognition_eye_blink()
        print(full_name, recognition_name)
        if full_name == recognition_name:
            flash('簽到成功', 'success')
        else:
            flash('簽到失敗', 'error')

    return render_template('home.html', full_name=full_name)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        department = request.form['department']
        id = request.form['id']
        password = request.form['password']
        gender = request.form['gender']
        full_name = request.form['full_name']
        email = request.form['email']
        phone = request.form['phone']
        address = request.form['address']
        print(department, id, password, gender, full_name, email, phone, address)

        # 連接到資料庫
        conn = connect_to_db()

        try:
            # 創建游標
            cursor = conn.cursor()

            # 新增資料
            cursor.execute(
                "INSERT INTO users (department, id, password, sex, full_name, email, phone, address, identity, su) VALUES (?, ?, ?, ?, ?, ?, ?, ?, '學生', 'F')",
                (department, id, password, gender, full_name, email, phone, address))

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
    global camera
    camera = cv2.VideoCapture(0)
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

@app.route('/video_capture')
def video_capture():
    return Response(capture_by_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/face_result')
def face_result():
    return Response(face_recognition_eye_blink_img(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

if __name__ == '__main__':
    app.run()