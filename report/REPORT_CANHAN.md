# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Bùi Thị Ngọc Trân
**Nhóm:** 4
**Vai trò trong so sánh nhóm:** Thành viên 4
**Chiến lược cá nhân:** `SentenceChunker(max_sentences_per_chunk=3)`
**Ngày:** 19/09/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> *Viết 1-2 câu:* Cosine similarity đo mức độ cùng hướng của hai vector. Với embedding có ngữ nghĩa, điểm cao thường cho thấy hai văn bản có ý nghĩa gần nhau, nhưng không bảo đảm chúng hoàn toàn đồng nghĩa.

**Ví dụ có độ tương tự CAO:**
- Câu A: Tôi muốn tạm dừng việc học một học kỳ
- Câu B: Em cần bảo lưu kết quả trong một kỳ
- Tại sao tương đồng: Hai câu đều thể hiện tạm nghỉ học dù dùng từ khác nhau

**Ví dụ có độ tương tự THẤP:**
- Câu A: Sinh viên phải đăng kí học phần
- Câu B: Chiếc bánh có vị sô cô la
- Tại sao khác: Hai câu nói về hai chủ đề khác nhau

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> *Viết 1-2 câu:* Cosine tập trung vào hướng vector và không phụ thuộc độ lớn, nên hữu ích khi hướng biểu diễn nội dung ngữ nghĩa. Khi các vector đã chuẩn hóa, cosine và Euclid cho thứ tự tương đồng tương đương.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* ceil((10000−50)/(500−50)) = ceil(9950/450) 
> *Đáp án:* 23

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> *Viết 1-2 câu:* Khi overlap tăng lên 100, số chunk là ceil((10000−100)/(500−100)) = 25. Overlap lớn hơn giúp giữ ngữ cảnh ở ranh giới chunk, nhưng làm tăng nội dung trùng lặp.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Chiến lược cá nhân và cấu hình so sánh của nhóm

| Thành viên | Chiến lược | Cấu hình đề xuất |
|---|---|---|
| Thành viên 1 — NgoGiaQuoc | `HeadingChunker` | `chunk_size=500`; section dài dùng `RecursiveChunker` |
| Thành viên 2 | `FixedSizeChunker` | `chunk_size=500, overlap=50` |
| Thành viên 3 | `RecursiveChunker` | `chunk_size=500`, separator mặc định |
| Thành viên 4 — Bùi Thị Ngọc Trân (tôi) | `SentenceChunker` | `max_sentences_per_chunk=3` |

Tôi sử dụng chiến lược thứ 4: chia văn bản theo ranh giới câu và nhóm tối đa 3 câu thành một chunk, không chồng lặp câu giữa các chunk. Tôi chọn cấu hình này với kỳ vọng giữ được câu trọn vẹn và ngữ cảnh liền kề khi truy xuất các hướng dẫn, quy định học vụ VinUni.

Giới hạn của cách làm là số câu không quyết định độ dài ký tự: các câu dài hoặc danh sách thiếu dấu kết thúc câu có thể tạo chunk lớn. Regex đơn giản cũng có thể tách chưa đúng với chữ viết tắt; các thông tin liên quan nằm ở hai nhóm câu khác nhau có thể bị tách rời. Đây là những điểm cần kiểm tra bằng kết quả benchmark, chưa phải kết luận về chất lượng truy xuất thực tế.

Cấu hình chạy cá nhân trong `bench.py`:

```python
CHUNKER = SentenceChunker(max_sentences_per_chunk=3)
TOP_K = 3
```

Tôi giữ nguyên 5 câu hỏi, đáp án chuẩn và điều kiện lọc của nhóm. Câu hỏi số 5 sử dụng `metadata_filter={"audience": "student"}`; benchmark còn in kết quả không lọc để đối chiếu. Khi so sánh chiến lược, các thành viên cần dùng cùng bộ tài liệu và cùng embedding backend.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> *Viết 2-3 câu: dùng biểu thức chính quy (regex) gì để phát hiện câu? Xử lý trường hợp ngoại lệ (edge case) nào?* Tôi dùng biểu thức chính quy (?<=[.!?])\s+ để tách văn bản tại khoảng trắng sau dấu kết thúc câu, đồng thời giữ lại dấu câu. Sau đó, nhóm tối đa 3 câu thành một chunk và loại bỏ các đoạn rỗng. Văn bản rỗng hoặc chỉ chứa khoảng trắng trả về danh sách rỗng.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> *Viết 2-3 câu: thuật toán hoạt động thế nào? Base case (trường hợp cơ sở) là gì?* Tôi tách văn bản theo thứ tự ưu tiên: đoạn văn, xuống dòng, câu, từ và cuối cùng là ký tự. Đệ quy dừng khi đoạn có độ dài không vượt quá chunk_size; nếu hết dấu phân cách thì cắt trực tiếp theo số ký tự. Các phần nhỏ được ghép lại nếu tổng độ dài vẫn không vượt quá giới hạn chunk_size.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> *Viết 2-3 câu: lưu trữ thế nào? Tính độ tương tự ra sao?* Tôi lưu mỗi chunk trong bộ nhớ cùng ID, nội dung, metadata và embedding; ID được tạo bằng bộ đếm để tránh ghi đè. Khi tìm kiếm, tôi tạo embedding của câu hỏi, tính tích vô hướng với embedding từng chunk, sắp xếp điểm giảm dần và lấy tối đa top_k kết quả. Với mock embedder hiện tại, các vector đã được chuẩn hóa về độ dài 1 nên tích vô hướng tương đương cosine similarity.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> *Viết 2-3 câu: lọc (filter) trước hay sau? Xóa bằng cách nào?* Tôi lọc metadata trước khi tính điểm tìm kiếm; một chunk phải thỏa mãn tất cả điều kiện trong bộ lọc. Khi xóa tài liệu, tôi loại bỏ tất cả bản ghi có metadata["doc_id"] tương ứng. Hàm trả về True nếu có bản ghi bị xóa và False nếu không tìm thấy.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> *Viết 2-3 câu: cấu trúc prompt? Cách đưa ngữ cảnh (inject context) vào thế nào?* Tôi truy xuất các chunk liên quan, đánh số và ghép vào prompt cùng câu hỏi. Prompt yêu cầu trả lời dựa trên ngữ cảnh, dẫn nguồn theo số đoạn và nói rõ khi thiếu thông tin; sau đó gọi llm_fn để tạo câu trả lời. Nếu không tìm được chunk nào, agent trả thông báo thiếu thông tin ngay. Hiện answer() sử dụng tìm kiếm không lọc metadata và gọi demo_llm để hiển thị bản xem trước prompt. Việc so sánh có và không có bộ lọc ở câu 5 được thực hiện riêng trong bench.py.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

Chạy toàn bộ bộ kiểm thử bằng `python -m pytest tests/ -v`. Trích kết quả từ log thực tế:

```text
(.venv) PS C:\Users\admin\OneDrive\Desktop\AI_20K\K4-Day07-BuiThiNgocTran-2A202602529> python -m pytest tests/ -v          
================================================== test session starts ===================================================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\admin\OneDrive\Desktop\AI_20K\K4-Day07-BuiThiNgocTran-2A202602529\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\admin\OneDrive\Desktop\AI_20K\K4-Day07-BuiThiNgocTran-2A202602529
plugins: anyio-4.15.1
collected 42 items                                                                                                        

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED                               [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                                        [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED                                 [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED                                  [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                                       [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED                       [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED                             [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED                              [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED                            [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                                              [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED                              [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                                         [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                                     [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                                               [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED                      [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED                          [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED                    [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED                          [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                                              [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED                                [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                                  [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                                        [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED                             [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED                               [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED                   [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                                [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                         [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                                        [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                                   [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                               [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED                          [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED                              [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                                    [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED                              [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED           [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED                         [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED                        [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED            [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED                       [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED                [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED      [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED          [100%]

=================================================== 42 passed in 0.28s ===================================================
(.venv) PS C:\Users\admin\OneDrive\Desktop\AI_20K\K4-Day07-BuiThiNgocTran-2A202602529> 
```

**Số lượng bài test vượt qua (pass):** 42 / 42.

| Nhóm kiểm thử | Số test đạt |
|---|---:|
| Cấu trúc dự án | 2 |
| Giao diện lớp và mock embedder | 2 |
| FixedSizeChunker | 7 |
| SentenceChunker | 4 |
| RecursiveChunker | 4 |
| EmbeddingStore — thêm và tìm kiếm | 8 |
| KnowledgeBaseAgent | 2 |
| compute_similarity | 4 |
| ChunkingStrategyComparator | 3 |
| Lọc metadata | 3 |
| Xóa tài liệu | 3 |
| **Tổng** | **42** |

Toàn bộ 42 bài kiểm thử đều đạt, không có bài thất bại. Trong đó, cả 4 bài kiểm thử của SentenceChunker đều đạt. Kết quả xác nhận mã nguồn đáp ứng bộ kiểm thử được cung cấp; chất lượng truy xuất và câu trả lời của agent được đánh giá riêng ở phần 5.

---


## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
| --- |------ | ----- | ------- | ------------ | ----- |
| 1   | Tối đa bao nhiêu tín chỉ được phép chuyển vào chương trình đại học tại VinUni? | Quy trình tham gia chương trình trao đổi sinh viên ra nước ngoài, gồm điều kiện, các bước nộp hồ sơ và cách quy đổi tín chỉ. | cao | 0.0645 | Chưa kết luận về ngữ nghĩa — dùng mock |
| 2   | Học phần cần đáp ứng mức tương đương nội dung và điểm tối thiểu nào để được xem xét chuyển đổi tín chỉ? | Nội dung về yêu cầu tiếng Anh đầu vào và các quốc gia được công nhận là sử dụng tiếng Anh. | thấp | 0.0159 | Chưa kết luận về ngữ nghĩa — dùng mock |
| 3   | Trên SIS cần thao tác thế nào để hoàn tất đăng ký môn và trạng thái nào xác nhận đăng ký thành công? | Nhấn “Add” rồi “Register” để hoàn tất. Đảm bảo trạng thái môn học là “Registered”. | cao | −0.0648 | Chưa kết luận về ngữ nghĩa — dùng mock |
| 4   | Yêu cầu bảng điểm hoặc thư xác nhận thường mất bao lâu và phí mỗi bản là bao nhiêu? | Nội dung về English Proficiency Variations and Conditional Admission. | thấp | 0.0629 | Chưa kết luận về ngữ nghĩa — dùng mock |
| 5   | Cần thực hiện những bước nào để đăng ký học phần? | Thêm vào giỏ và đăng ký: Nhấn “Add” rồi “Register” để hoàn tất và kiểm tra trạng thái “Registered”.| cao | −0.0296 | Chưa kết luận về ngữ nghĩa — dùng mock |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**

> Kết quả bất ngờ nhất là cặp 3 và 5: nội dung có quan hệ hỏi–đáp rõ ràng nhưng điểm mock lần lượt là −0.0648 và −0.0296. Mock embedder tạo vector từ mã băm, không học ngữ nghĩa, nên điểm âm không chứng minh hai câu trái nghĩa. Tôi nhận ra cần phân biệt phép đo trên vector giả lập với khả năng biểu diễn ý nghĩa của embedding thực.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Tối đa bao nhiêu tín chỉ được phép chuyển vào chương trình đại học tại VinUni? | Quy trình trao đổi sinh viên ra nước ngoài, có đề cập quy đổi tín chỉ. | 0.2600 | Gần chủ đề nhưng không nêu giới hạn tín chỉ. | Demo hiển thị phần đầu prompt và đoạn về trao đổi sinh viên; chưa trả lời số tín chỉ. |
| 2 | Học phần cần đáp ứng mức tương đương nội dung và điểm tối thiểu nào để được xem xét chuyển đổi tín chỉ? | Các quốc gia sử dụng tiếng Anh và mục Prior study trong chính sách tiếng Anh đầu vào. | 0.3032 | Không liên quan trực tiếp đến điều kiện chuyển đổi tín chỉ. | Demo hiển thị phần đầu prompt và đoạn về quốc gia nói tiếng Anh; chưa trả lời điều kiện chuyển đổi. |
| 3 | Trên SIS cần thao tác thế nào để hoàn tất đăng ký môn và trạng thái nào xác nhận đăng ký thành công? | Điều kiện về nơi học trước đây trong chính sách tiếng Anh đầu vào. | 0.2270 | Không. | Demo hiển thị phần đầu prompt và đoạn về nơi học; chưa trả lời thao tác hoặc trạng thái đăng ký. |
| 4 | Yêu cầu bảng điểm hoặc thư xác nhận thường mất bao lâu và phí mỗi bản là bao nhiêu? | Danh sách và giới hạn học phần trong chính sách tiếng Anh đầu vào. | 0.2642 | Không. | Demo hiển thị phần đầu prompt và đoạn về học phần; chưa trả lời thời gian xử lý hoặc phí. |
| 5 | Cần thực hiện những bước nào để đăng ký học phần? | Các lỗi thường gặp khi đăng ký học phần. | 0.2073 | Top-1 liên quan chủ đề nhưng không nêu quy trình; hạng 2 chứa bước Add → Register và kiểm tra Registered. | Demo hiển thị phần đầu prompt và đoạn về lỗi đăng ký; chưa tổng hợp các bước. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 1/5 câu có chunk trực tiếp trả lời ít nhất một phần câu hỏi. Ở câu 5, chunk hạng 2 có score 0.1861, hướng dẫn nhấn “Add”, “Register” và kiểm tra trạng thái “Registered”, nhưng chưa bao gồm toàn bộ quy trình. Câu 3 truy xuất đúng tài liệu nguồn ở hạng 2, song đoạn đó chỉ nói về mốc Add/Drop, chưa trả lời thao tác và trạng thái đăng ký.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> *Viết 2-3 câu:* Tôi nhận ra cần cân nhắc giữa giữ ngữ cảnh rộng và chia theo câu để giữ từng ý trọn vẹn. Với SentenceChunker của tôi, câu 5 tìm được một phần hướng dẫn đăng ký ở hạng 2, nhưng câu 4 chưa tìm được thông tin thời gian và lệ phí; bộ lọc metadata cũng chưa làm thay đổi top-3. Tôi học được rằng cần đối chiếu nội dung chunk và dùng cùng embedding backend khi so sánh, thay vì kết luận chiến lược nào tốt hơn chỉ từ điểm số hoặc nhận xét chung.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5/ 5 |
| Hướng tiếp cận của tôi (My Approach) | 10/ 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5/ 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10/ 10 |
| **Tổng phần cá nhân** | **60/ 60** |
