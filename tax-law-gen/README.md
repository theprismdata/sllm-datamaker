## 원본 데이터 
### 판례 형식
데이터 형식
"index": 데이터 인덱스 (INT),
            "title": "제목"(STR) ,
            "summary": "주요내용 요약"(STR),
            "response": "답변사항"(STR),
            "content": "세부사항 전문(STR)",
            "metadata": {
                "date": "날짜"(STR),
                "answer_type": "해석사례 유형"(STR),
                "law_category": "세법 해당 카테고리"(STR),
                "department": "답변 부서 코드"(STR),
                "related_docs": [관련 문서](STR LIST),  
                "related_topics": "관련 주제"(STR),
                "related_laws": [관련 법](STR LIST)
            }
---------------------------------------------------------------------------
데이터 유형
JSON
---------------------------------------------------------------------------
데이터 종류
data_main: 날짜별로 순차적으로 스크래핑된 데이터(약 3만건)
data_view: 조회수별로 순차적으로 스크래핑된 데이터(1만건)
data_imp: 국세법령정보시스템에서 "주요 해석사례"로 채택되어 있는 데이터(117건)
verification_data

실제 사용 데이터는 data_main로 함.

