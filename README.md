Phân Loại Chó Mèo
=================================================

Ứng dụng web đơn giản này cho phép bạn phân loại hình ảnh chó hoặc mèo bằng mô hình học sâu ResNet50 mạnh mẽ và giao diện người dùng trực quan được xây dựng với Streamlit.

Công nghệ sử dụng
---------------------
* Python (phiên bản 3.8 trở lên)
* Streamlit
* PyTorch
* Torchvision
* Pillow (PIL)

Cài đặt và Chạy ứng dụng
-----------------------------
1. Tạo và kích hoạt môi trường ảo (khuyến khích):

   * **Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

    * **macOS/Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```


2. Cài đặt các thư viện cần thiết:

   Chạy lệnh sau trong terminal:
   > pip install -r requirements.txt

3. Chuẩn bị mô hình:
   - Đặt tệp `checkpoint_epoch_10.pt` đã tải vào thư mục `model/`.

4. Chạy ứng dụng:
   Mở terminal và điều hướng đến thư mục gốc của dự án, sau đó chạy lệnh:
   > streamlit run app.py

5. Truy cập ứng dụng:
   Mở trình duyệt web của bạn và truy cập vào địa chỉ: http://localhost:8501

Chi tiết về Mô hình
-----------------------
* Kiến trúc: ResNet50, một mạng nơ-ron tích chập sâu nổi tiếng, đã được huấn luyện trước trên bộ dữ liệu ImageNet khổng lồ.
* Tinh chỉnh (Fine-tuning): Chỉ có lớp kết nối đầy đủ (fully connected layer) cuối cùng của mô hình được tinh chỉnh lại cho nhiệm vụ phân loại chó và mèo.
* Đầu ra: Mô hình thực hiện phân loại nhị phân:
    * 0: Mèo 🐱
    * 1: Chó 🐶

Thiết kế Giao diện Người dùng (UI)
-------------------------------------
* Tải ảnh lên: Hỗ trợ các định dạng ảnh phổ biến như JPG, JPEG, và PNG.
* Bố cục:
    * Hình ảnh đã tải lên được hiển thị ở phía bên trái của giao diện.
    * Kết quả dự đoán (Chó hoặc Mèo) cùng với biểu tượng cảm xúc tương ứng được hiển thị ở phía bên phải.