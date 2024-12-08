import streamlit as st
from function import *

st.title('HỆ THỐNG DỰ ĐOÁN GIÁ NHÀ')
st.write('Đây là ứng dụng dự đoán giá nhà sử dụng dữ liệu từ bộ dữ liệu Boston Housing.')

name = st.text_input('Nhập tên của bạn:')
st.write('Chào mừng bạn đến với ứng dụng của chúng tôi,', name)
age = st.slider('Nhập tuổi của bạn:', 0, 100, 30)

square_feet = st.number_input('Nhập diện tích nhà (feet vuông):', 100)
bedrooms = st.number_input('Nhập số phòng ngủ:', 1)
bathrooms = st.number_input('Nhập số phòng tắm:', 1)
neighborhood = st.selectbox('Chọn khu vực:', ['Rural', 'Suburb', 'Urban'])
year_built = st.number_input('Nhập năm xây dựng:', 1900)


if st.button('Dự đoán giá nhà'):
    st.write('Đang dự đoán...')
    # Load the model
    lr_model, nn_model, encoder, scaler = load_file()
    # Preprocess
    preprocessed_data = preprocess(square_feet, bedrooms, bathrooms, neighborhood, year_built, encoder, scaler)
    # Make prediction
    prediction = predict(preprocessed_data, lr_model)
    st.write('Giá nhà dự đoán:', prediction, '$')