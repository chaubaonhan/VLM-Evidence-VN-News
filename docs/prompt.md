# Classification prompt

The prompt below is produced by `vlmstudy.core.classify_prompt` and used verbatim for both models and in
every input condition. It is sent as a single user turn of each model's chat template, with the generation
prompt appended. When an image is given, the image placeholder precedes the text inside the same user
turn; it is omitted in the text-only condition. `{text}` is replaced by the article text truncated to
1,024 tokens, or by `(Không có văn bản)` when the text is absent (image-only condition).

Label probabilities are read from the next-token logits at the first position of the assistant turn,
restricted to the first token of each of the ten English label names (constrained scoring).

```text
Bạn là hệ thống phân loại chủ đề tin tức tiếng Việt. Hãy xác định chủ đề của bài báo dựa trên văn bản bài báo và ảnh minh họa đi kèm (nếu có).

Danh sách nhãn:
- Automotive: Xe – ô tô, xe máy, thị trường xe, giao thông
- Business: Kinh doanh – doanh nghiệp, tài chính, chứng khoán, thị trường
- Education: Giáo dục – trường học, thi cử, tuyển sinh, du học
- Entertainment: Giải trí – nghệ sĩ, phim ảnh, âm nhạc, hoa hậu, thời trang
- Health: Sức khỏe – bệnh tật, y tế, dinh dưỡng, chăm sóc sức khỏe
- Law: Pháp luật – vụ án, điều tra, xét xử, tư vấn pháp lý
- Lifestyle: Đời sống – gia đình, quan hệ, nhịp sống, câu chuyện đời thường
- Politics: Thời sự – chính trị, chính sách, hoạt động nhà nước, sự kiện trong nước
- Sports: Thể thao – bóng đá, giải đấu, vận động viên
- World: Thế giới – tin quốc tế, quan hệ giữa các nước

Văn bản bài báo:
"""
{text}
"""

Chỉ trả lời bằng đúng một nhãn tiếng Anh trong danh sách, không giải thích.
```

English translation: "You are a Vietnamese news topic classification system. Determine the topic of the
article based on the article text and the accompanying illustration (if any). Label list: [the ten labels
with Vietnamese glosses]. Article text: [{text}]. Answer with exactly one English label from the list,
without explanation."

The OCR-only condition (text read from the photograph, no article text and no image) uses the same prompt,
with `{text}` replaced by the prefix `(Chữ nhận dạng được trong ảnh, không phải nội dung bài báo) ` ("Text
recognised in the image, not the article content") followed by the OCR string; see `src/vlmstudy/ocr.py`.
