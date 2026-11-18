# Dental OCT air-background correction

AI 없이 밝기 경계로 공기층과 치아 표면을 추정하고, 열별 공기층 중앙값으로 세로 방향 밝기 편차를 보정하는 실험용 Python 모듈입니다.

## 설치

Python 3.10 이상에서:

```sh
python -m pip install -r requirements.txt
```

## 프로그램 연동

```python
from oct_air_correction_70 import correct_oct, contrast_uint8

# frame: 원본 grayscale NumPy 배열, uint8, shape=(1024, 500)
result = correct_oct(frame)
corrected_frame = result.corrected
print(result.status)

# 선택적 표시용 대비 조정. 45..112는 예시 영상에 사용한 값입니다.
contrast_frame = contrast_uint8(result.corrected_float, black=45, white=112)
```

입력 배열은 변경하지 않으며, 파일 저장이나 GUI 처리는 호출하는 프로그램에서 담당합니다.

## 보정 설정

- 밝기 차이와 연속성 제약으로 열별 표면을 자동 추정합니다.
- 각 열에서 20행부터 표면 20픽셀 위까지의 원본 픽셀 중앙값을 계산합니다.
- 공기층 보정 강도는 100%, 치아 내부는 70%입니다.
- 표면 아래 60픽셀 동안 강도를 100%에서 70%로 부드럽게 줄입니다.
- 신뢰 조건을 통과하지 못한 열은 보정하지 않습니다. 중앙 영역에 유효 열이 없으면 전체 원본을 반환합니다.
- `status`는 `ok`, `partial_surface`, `no_valid_surface` 중 하나입니다. `ok`는 휴리스틱 통과를 의미하며 정확한 경계를 보장하지 않습니다.

수식과 상세 구현은 `oct_air_correction_70.py`의 한글 설명을 참고하세요.

## 제한 사항

현재 1024×500의 8bit 표시 영상에서 확인한 실험용 알고리즘입니다. 원시 OCT 간섭 신호나 16bit 영상에 직접 적용하지 마세요. 다른 해상도·장비·촬영 조건에서는 경계 검출과 보정 설정을 재검증해야 합니다. 잔여 줄무늬와 조직 밝기 왜곡이 생길 수 있으며, 대비 조정은 약한 신호를 검정으로 잘라낼 수 있습니다. 임상적 정확도와 실시간 처리 성능은 검증되지 않았습니다.

저장소에는 환자 영상이나 데이터셋을 포함하지 않습니다.
