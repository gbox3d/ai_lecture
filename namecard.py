import cv2
import pytesseract

def extract_text(frame):
    # 이미지를 회색조로 변환합니다.
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 이미지에서 텍스트를 추출합니다.
    text = pytesseract.image_to_string(gray_image, lang='kor')

    return text

# 카메라를 초기화합니다.
cap = cv2.VideoCapture(0)

while True:
    # 카메라에서 프레임을 읽습니다.
    ret, frame = cap.read()

    # 프레임에서 텍스트를 추출합니다.
    extracted_text = extract_text(frame)

    # 추출된 텍스트를 출력합니다.
    print("Extracted Text:", extracted_text)

    # 프레임을 화면에 표시합니다.
    cv2.imshow('frame', frame)

    # 'q' 키를 누르면 루프를 종료합니다.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라를 해제하고, 모든 창을 닫습니다.
cap.release()
cv2.destroyAllWindows()
