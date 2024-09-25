import pandas as pd

# Đọc dữ liệu từ CSV
df = pd.read_csv('D:/data_hocmay/databangoc.csv')

# Kiểm tra xem cột 'Số thứ tự' đã tồn tại chưa
if 'STT' not in df.columns:
    df.insert(0, 'STT', range(1, len(df) + 1))

# Lưu dữ liệu vào CSV
df.to_csv('D:/data_hocmay/mynewdata.csv', index=False)

