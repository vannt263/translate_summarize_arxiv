# Dữ liệu:
https://www.kaggle.com/code/sivadhas/preprocessing/input
Dung lượng: 1.2GB
Với hai bộ dữ liệu được lấy từ 2 kênh tin tức lớn của Mỹ là CNN và Daily Mail":
- CNN: https://edition.cnn.com/
- DailyMail: https://www.dailymail.co.uk/home/index.html

Bộ dữ liệu CNN/Daily Mail là một tập dữ liệu được sử dụng để tạo tóm tắt cho văn bản. Những đoạn tóm tắt này được người ta tạo ra từ các bài báo trên trang web của CNN và Daily Mail. 
--------------------------------------------------------------------------------------------------------------
# Dữ liệu cho việc tóm tắt bao gồm:
Id: ID bản tin.
Article: Nội dung bản tin.
Highlights: Nội dung tóm tắt của bản tin

Sử dụng 100000 dòng (tương ứng với 100000 bản tin được lấy từ 2 kênh thông tin lớn: CNN và DailyMail). 
-> processing lưu trữ dưới dạng data.csv
-----------------------------------------------------------------------------------------------------------------
# Dữ liệu cho việc dịch tiếng việt bao gồm:
Thực hiện gán nhãn vi cho 1000 file cho mục đích fine tuning model dịch văn bản:
Sử dụng module Translator.
Thực hiện gán nhãn bằng tay.
# File lưu trữ model:
https://drive.google.com/drive/folders/1WFnB_WyxlU-s8ipStQFau30JMJuzyXwc?usp=sharing
-----------------------------------------------------------------------------------------------------------
# GloVe dùng để Embedding: 
https://nlp.stanford.edu/projects/glove/
----------------------------------------------------------------------------------------------------------
# Cấu trúc folder



                                                        |---- 01_select_file.ipynb: Chọn file phù hợp cho quá trình làm project
                                                        |
                                                        |---- 02_preprocessing.ipynb: tiền xử lý dữ liệu
                                                        |
                                                        |---- 03_abstract_summary_v1.ipynb: model thứ 1 để train mô hình seq2seq
                                                        |
                                                        |---- 03_abstract_symmary.ibynb: model thứ 2 để train mô hình seq2seq
                                                        |
                                                        |---- 03_extractive_summary.ipynb: model gồm 2 thuật toán trích xuất
                                                        |
                                                        |---- 04_finetuning_translate.ipynb: model để dịch dữ liệu
                                    ----------> code ---|
                                    |
                                    |
                                    |----------> data_summarize: gồm các file txt cho quá trính tóm tắt sau khi đã loại bỏ trường id
                                    |
                                    |----------> data_translate: gồm các file txt cho quá trình dịch sau khi thêm trường vi
translate_summarie_news ------------|
                                    |----------> demo -- app.py: chạy sản phẩm
                                    |
                                    |----------> README.md
                                    |
                                    |----------> requirement -- tải các thư viện để chạy project







