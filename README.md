Dữ liệu gốc được dowload từ https://huggingface.co/datasets/ccdv/arxiv-summarization. Với một số các thuộc tính:
- Article ID: ID của bài báo.
- Article Text: Nội dung của bài báo.
- Abstract Text: Nội dung tóm tắt của bài báo.
- Labels: Nhãn của bài báo.
- Section name: Các mục có trong bài báo.
- Section: Nội dung cụ thể của các mục trong bài báo.

Phần nội dung Article Text là tổng hợp lại của các Section.
Trong project này, chúng tôi chỉ tập trung sử dụng vào hai trường chính là: Article Text và Abstract Text, nhóm sử dụng transform_data.ipynb để bóc tách dữ liệu.