import cv2
import easyocr

# Khởi tạo đối tượng EasyOCR với ngôn ngữ muốn nhận dạng
reader = easyocr.Reader(['en'])  # Thay 'en' bằng ngôn ngữ bạn muốn sử dụng

# Đường dẫn đến file video
video_path = 'C:/Users/TuanBao/Desktop/My_Docs/Project_Atin/Project_Atin/Project_team2/video/video3.mp4'

# Khởi tạo đối tượng VideoCapture
cap = cv2.VideoCapture(video_path)

cap = cv2.VideoCapture(video_path)

# Đọc frame đầu tiên từ video
ret, frame = cap.read()

if ret:
    # Chuyển đổi khung hình sang ảnh grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Áp dụng EasyOCR để nhận dạng văn bản trên khung hình
    output = reader.readtext(gray)

    # Hiển thị kết quả nhận dạng văn bản trên khung hình
    for result in output:
        top_left = tuple(result[0][0])
        bottom_right = tuple(result[0][2])
        text = result[1]
        confidence = result[2]

        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(frame, f'{text} ({confidence:.2f})', top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        print(top_left, bottom_right, text, confidence)
    # Hiển thị khung hình kết quả
    # cv2.imshow('First Frame', frame)
    # cv2.waitKey(0)

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()