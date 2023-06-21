import cv2
import numpy as np
import dlib
from PIL import ImageDraw, ImageFont, Image
from db import connect_to_db

def face_recognition_eye_blink(camera):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('train/train.yml')
    cascadePath = "model_landmarks/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    minW = 0.1 * camera.get(3)
    minH = 0.1 * camera.get(4)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("model_landmarks/shape_predictor_68_face_landmarks.dat")

    def eye_aspect_ratio(eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    ear_thresh = 0.2  # 0.225
    ear_consec_frames = 3  # 2
    COUNTER = 0
    detected_name = None  # 儲存偵測到的姓名
    max_frames_without_detection = 300  # 設定最大連續幀數沒有偵測到眨眼和姓名(大約30秒)
    frame_counter = 0

    conn = connect_to_db()  # 連接資料庫
    cursor = conn.cursor()

    def get_user_name(user_id):
        query = "SELECT full_name FROM users WHERE id = ?"
        cursor.execute(query, (user_id,))
        result = cursor.fetchone()
        if result:
            return result[0]
        else:
            return "None"

    while True:
        ret, img = camera.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = np.array([(p.x, p.y) for p in shape.parts()])

            left_eye = shape[36:42]
            right_eye = shape[42:48]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            ear = (left_ear + right_ear) / 2.0

            if ear < ear_thresh:
                COUNTER += 1

                if COUNTER >= ear_consec_frames:
                    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))
                    for (x, y, w, h) in faces:
                        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                        if confidence < 50:
                            name = get_user_name(id)
                            detected_name = name  # 更新偵測到的姓名
                            # camera.release()  # 釋放攝像頭資源
                            cursor.close()
                            conn.close()
                            return detected_name  # 返回偵測到的姓名
                        else:
                            name = "None"

            frame_counter += 1
            print(frame_counter)
            if frame_counter >= max_frames_without_detection:
                # camera.release()  # 釋放攝像頭資源
                cursor.close()
                conn.close()
                return "水喔"


def face_recognition_eye_blink_img(camera):
    recognizer_img = cv2.face.LBPHFaceRecognizer_create()
    recognizer_img.read('train/train.yml')
    cascadePath = "model_landmarks/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    minW = 0.1 * camera.get(3)
    minH = 0.1 * camera.get(4)
    font_size = 30
    font = ImageFont.truetype('font/NotoSansTC-Bold.otf', font_size)

    detector_img = dlib.get_frontal_face_detector()
    predictor_img = dlib.shape_predictor("model_landmarks/shape_predictor_68_face_landmarks.dat")

    def eye_aspect_ratio(eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    ear_thresh = 0.2  # 0.225
    ear_consec_frames = 3  # 2
    COUNTER = 0

    conn = connect_to_db()  # 連接資料庫
    cursor = conn.cursor()

    def get_user_name(user_id):
        query = "SELECT full_name FROM users WHERE id = ?"
        cursor.execute(query, (user_id,))
        result = cursor.fetchone()
        if result:
            return result[0]
        else:
            return "未知"

    while True:
        ret, img = camera.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector_img(gray, 0)

        # 在迴圈之外初始化一個空的圖片
        img_with_overlay = img.copy()

        for rect in rects:
            (x, y, w, h) = rect.left(), rect.top(), rect.width(), rect.height()
            cv2.rectangle(img_with_overlay, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            shape = predictor_img(gray, rect)
            shape = np.array([(p.x, p.y) for p in shape.parts()])

            left_eye = shape[36:42]
            right_eye = shape[42:48]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            ear = (left_ear + right_ear) / 2.0

            if ear < ear_thresh:
                COUNTER += 1

                if COUNTER >= ear_consec_frames:
                    # 將中文字繪製到影像中
                    imgPil = Image.fromarray(cv2.cvtColor(img_with_overlay, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(imgPil)
                    text = "偵測到眨眼"
                    draw.text((10, 10), text, fill=(255, 0, 0), font=font)
                    img_with_overlay = cv2.cvtColor(np.array(imgPil), cv2.COLOR_RGB2BGR)
            else:
                COUNTER = 0

            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5,
                                                 minSize=(int(minW), int(minH)))
            for (x, y, w, h) in faces:
                cv2.rectangle(img_with_overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
                id, confidence = recognizer_img.predict(gray[y:y + h, x:x + w])
                if confidence < 50:
                    name_img = get_user_name(id)
                    confidence = str(100 - round(confidence)) + "%"
                else:
                    name_img = "未知"
                    confidence = str(100 - round(confidence)) + "%"

                # 將中文字繪製到影像中
                imgPil = Image.fromarray(cv2.cvtColor(img_with_overlay, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(imgPil)
                text = name_img + " " + confidence
                draw.text((x + 5, y + h + 20), text, fill=(0, 255, 0), font=font)
                img_with_overlay = cv2.cvtColor(np.array(imgPil), cv2.COLOR_RGB2BGR)

        ret, buffer = cv2.imencode('.jpg', img_with_overlay)
        if not ret:
            continue  # 跳過圖像編碼錯誤

        img_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')
