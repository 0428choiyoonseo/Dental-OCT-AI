"""OCT 공기층 기반 열 보정: 공기 100%, 치아 내부 70% (AI 없음).

의존성: numpy, scipy. 파일 I/O 및 GUI 의존성 없음.
입력: shape=(1024, 500), dtype=uint8인 원본 grayscale 표시 영상.
원시 간섭 신호/16bit 영상에 직접 적용하지 말 것. 입력 배열은 변경하지 않음.
현재 해상도에서 확인한 실험 알고리즘이며 임상/실시간 성능은 검증되지 않음.

사용 예:
    from oct_air_correction_70 import correct_oct, contrast_uint8
    result = correct_oct(frame)
    corrected_frame = result.corrected
    # 선택적 표시용 대비 조정. 45..112는 이번 예시 영상의 창 값임.
    contrast_frame = contrast_uint8(result.corrected_float, black=45, white=112)
    # result.status 및 result.valid_columns로 경계 검출 실패 확인.

수식 (z=행/깊이, x=열, 모든 좌표 0-based):
    J = Gaussian(I, sigma=(2,2))  # 경계 검출용으로만 사용
    D(z,x) = mean(J[z:z+16,x]) - mean(J[z-16:z,x])
    s = argmax sum D(s[x],x) - 0.15*sum |s[x]-s[x-1]|
        단 80 <= s < H-80, 인접 열 깊이 차이 <=12; 이후 5열 median
    b[x] = median(I[20:s[x]-20,x])
    b0 = median(유효한 b[10:W-10])
    u = clip((z-s[x])/60, 0, 1)
    alpha = 1 - 0.3*(3*u**2 - 2*u**3)
    C = I - alpha*(b[x]-b0)

공기층은 alpha=1, 표면 아래 60픽셀 동안 1→0.7, 이후 0.7.
검출 점수<5 또는 공기 픽셀<40인 열은 보정하지 않음.
중앙 유효 열이 없으면 전체 원본을 반환하고 status='no_valid_surface'.
검출 점수는 휴리스틱이며 해부학적 정확도를 보장하지 않음.
표시용 대비는 C의 반올림/클리핑 전 값을 사용:
    round(255*clip((C-black)/(white-black),0,1))
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter


@dataclass
class CorrectionResult:
    corrected: np.ndarray
    corrected_float: np.ndarray
    surface_rows: np.ndarray
    valid_columns: np.ndarray
    edge_scores: np.ndarray
    column_offsets: np.ndarray
    background_reference: float | None
    status: str


def _uint8(values):
    return np.rint(np.clip(values, 0, 255)).astype(np.uint8)


def _surface(a):
    j = gaussian_filter(a, sigma=(2, 2))
    n = 16
    cs = np.pad(np.cumsum(j, axis=0), ((1, 0), (0, 0)))
    depths = np.arange(80, a.shape[0] - 80)
    score = ((cs[depths+n] - cs[depths]) / n
             - (cs[depths] - cs[depths-n]) / n)
    count, width = score.shape
    back = np.zeros((width, count), dtype=np.int32)
    prev = score[:, 0].copy()
    for x in range(1, width):
        best = np.full(count, -np.inf)
        parent = np.zeros(count, dtype=np.int32)
        for d in range(-12, 13):
            lo, hi = max(0, -d), min(count, count-d)
            candidate = prev[lo+d:hi+d] - .15 * abs(d)
            update = candidate > best[lo:hi]
            indices = np.arange(lo, hi)
            best[indices[update]] = candidate[update]
            parent[indices[update]] = indices[update] + d
        prev = best + score[:, x]
        back[x] = parent
    path = np.zeros(width, dtype=int)
    path[-1] = np.argmax(prev)
    for x in range(width-1, 0, -1):
        path[x-1] = back[x, path[x]]
    surface = median_filter(depths[path], size=5, mode='nearest')
    confidence = score[surface-depths[0], np.arange(width)]
    return surface, confidence


def correct_oct(image: np.ndarray) -> CorrectionResult:
    """원본 uint8 프레임을 보정. 입력/출력 shape=(1024,500).

    잘못된 입력은 예외 발생. 검출 실패 시에는 보정 없이 원본 반환.
    해상도를 자동 resize하지 않음: 다른 장비의 좌표/스케일은 재검증 필요.
    """
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.shape != (1024, 500):
        raise ValueError('Expected grayscale uint8 image with shape (1024, 500).')
    a = image.astype(np.float64)
    surface, confidence = _surface(a)
    valid = (surface-20-20 >= 40) & (confidence >= 5)
    offset = np.zeros(a.shape[1], dtype=np.float64)
    if not np.any(valid[10:-10]):
        return CorrectionResult(image.copy(), a, surface, valid, confidence,
                                offset, None, 'no_valid_surface')
    b = np.full(a.shape[1], np.nan)
    for x in np.flatnonzero(valid):
        b[x] = np.median(a[20:surface[x]-20, x])
    b0 = float(np.nanmedian(b[10:-10]))
    offset[valid] = b[valid] - b0
    u = np.clip((np.arange(a.shape[0])[:, None] - surface[None, :]) / 60, 0, 1)
    alpha = 1 - .3 * (3*u*u - 2*u*u*u)
    corrected = a - alpha * offset[None, :]
    status = 'ok' if np.all(valid) else 'partial_surface'
    return CorrectionResult(_uint8(corrected), corrected, surface, valid,
                            confidence, offset, b0, status)


def contrast_uint8(corrected_float: np.ndarray, *, black: float,
                   white: float) -> np.ndarray:
    """선택적 표시용 대비. 약한 신호가 검정으로 잘릴 수 있음.

    black/white는 보정 계수가 아님. 동일 비교에는 동일 창 값을 사용.
    이번 이미지 재현값은 black=45, white=112. 모든 영상의 최적값은 아님.
    """
    a = np.asarray(corrected_float, dtype=np.float64)
    if a.ndim != 2 or not np.all(np.isfinite(a)):
        raise ValueError('Expected a finite 2D image.')
    if not np.isfinite(black) or not np.isfinite(white) or white <= black:
        raise ValueError('Contrast window must be finite with white > black.')
    return _uint8(255 * np.clip((a-black) / (white-black), 0, 1))
