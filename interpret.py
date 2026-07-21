#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
규칙기반 해석기 (rule-based interpreter)
-----------------------------------------
meow_analyze.py 가 만든 분석 JSON 을 받아, 사람이 손으로 하던 판정 보정을
자동으로 적용해서 top_type / summary / 건강 메모를 만들어 낸다.

핵심 보정 규칙(그동안 수동으로 적용하던 것):
  1) 유성비율이 매우 낮고(<0.25) 짧은(<0.25s) 구간은 '환경음/오검출'로 보고 대표 유형에서 제외
  2) 'pain/distress(통증)' 후보는 유성비율<0.5 일 때 F0 오추적 아티팩트로 보고 무시
  3) 상황 메모(context_note)를 최우선 단서로 사용해 대표 유형을 좁힘
  4) 진짜 발성 구간들에서 지배적 유형을 골라 대표 유형으로 삼음
  5) 건강 신호(쉰 목소리 등)는 모으되, 단일 클립 진단은 하지 않음 (관찰 메모 수준)
사용법:
  python interpret.py analysis.json  ->  같은 파일에 top_type/summary/interpretation 추가
"""
import sys, json, re

# ----------------------------------------------------------------- 상황→유형 힌트
CONTEXT_HINTS = [
    (r"출입|나가|문|베란다|창",      "요구형 야옹(출입)",      "닫힌 공간 앞 출입 요구"),
    (r"밥|사료|간식|먹이|배고",       "먹이 요구",             "먹이·간식 요구"),
    (r"아침|기상|일어",              "아침 인사·애정 요구",    "아침 기상 인사"),
    (r"헤딩|부비|비비|범핑|bunting", "애정·관심 요구",         "몸 부비기 동반 애정 표현"),
    (r"놀|장난감",                   "놀이 요구",             "놀이·관심 요구"),
    (r"호출|불러|다가|왔|이리",       "호출·접촉 요구",         "사람을 부르는 접촉 요구"),
    (r"인사|반가|맞이",              "인사·관심",             "인사·관심 표현"),
    (r"낮|잠깐|스쳐|지나",           "느긋한 인사",           "가벼운 낮 인사"),
]

def _norm_type(t):
    """후보 type 문자열을 대분류 키로 축약."""
    if not t: return "기타"
    if "food" in t or "solicitation" in t or "먹이" in t or "요구" in t: return "요구"
    if "trill" in t or "트릴" in t or "chirrup" in t: return "트릴·인사"
    if "stress" in t or "불편" in t: return "불편"
    if "complaint" in t or "불만" in t: return "불만·항의"
    if "yowl" in t or "caterwaul" in t: return "장음"
    if "growl" in t or "그르렁" in t or "purr" in t: return "저음·그르렁"
    if "hiss" in t or "하악" in t: return "하악"
    if "pain" in t or "통증" in t: return "통증"
    if "meow" in t or "야옹" in t: return "일반 야옹"
    return "기타"


def is_false_detection(m):
    """환경음/오검출로 볼 구간인지."""
    vr = m.get("voiced_ratio") or 0
    dur = m.get("duration_s") or 0
    hnr = m.get("hnr_db")
    flat = m.get("spectral_flatness") or 0
    if vr < 0.25 and dur < 0.30:
        return True
    if vr < 0.20 and (hnr is not None and hnr < -3) and flat > 0.20:
        return True
    return False


def best_candidate(seg):
    """통증 아티팩트 등을 걸러낸 뒤의 대표 후보."""
    m = seg["metrics"]
    vr = m.get("voiced_ratio") or 0
    cands = list(seg.get("candidates") or [])
    filtered = []
    for c in cands:
        t = c["type"]
        # 통증 후보: 유성비율 낮으면 F0 오추적 아티팩트로 간주하여 제외
        if ("pain" in t or "통증" in t) and vr < 0.5:
            continue
        # 하악 후보: 유성 성분이 거의 없고 매우 짧으면 환경음일 수 있어 감점
        filtered.append(c)
    if not filtered:
        filtered = cands
    filtered = sorted(filtered, key=lambda c: -c.get("score", 0))
    return filtered[0] if filtered else None


def collect_health(segments):
    """유성 구간의 실제 건강 신호만 모은다(오검출 구간 제외)."""
    notes = []
    hoarse = 0
    for s in segments:
        m = s["metrics"]
        if is_false_detection(m):
            continue
        for f in s.get("health_flags", []):
            fl = f["flag"]
            if "쉰 목소리" in fl:
                hoarse += 1
            notes.append((s["segment"], fl))
    return notes, hoarse


def interpret(data):
    segs = data.get("segments", [])
    ctx = (data.get("context_note") or "").strip()

    genuine, noise = [], []
    for s in segs:
        (noise if is_false_detection(s["metrics"]) else genuine).append(s)

    # 대표 후보 유형 집계 (진짜 발성만)
    type_counts, clean_meows, rough = {}, [], []
    f0_solicit = []
    for s in genuine:
        bc = best_candidate(s)
        key = _norm_type(bc["type"] if bc else None)
        type_counts[key] = type_counts.get(key, 0) + 1
        m = s["metrics"]
        hnr = m.get("hnr_db")
        vr = m.get("voiced_ratio") or 0
        if hnr is not None and hnr >= 4 and vr >= 0.7:
            clean_meows.append(s["segment"])
        if hnr is not None and hnr < 2 and vr > 0.4:
            rough.append(s["segment"])
        if key == "요구" and m.get("f0_mean_hz"):
            f0_solicit.append(m["f0_mean_hz"])

    # ---- 대표 유형 결정: 상황 메모 우선, 없으면 최빈 유형
    top_type, ctx_label = None, None
    for pat, label, note in CONTEXT_HINTS:
        if ctx and re.search(pat, ctx):
            top_type, ctx_label = label, note
            break
    if not top_type:
        if type_counts:
            dom = max(type_counts, key=type_counts.get)
            top_type = {"요구": "요구형 야옹", "트릴·인사": "트릴·인사",
                        "일반 야옹": "일반 야옹", "저음·그르렁": "저음·그르렁 인사",
                        "장음": "장음(발정·불안 감별)", "불만·항의": "불만·항의",
                        "불편": "불편·스트레스", "하악": "경계·불쾌"}.get(dom, dom)
        else:
            top_type = "분류 보류"

    # ---- 건강 메모
    health_notes, hoarse = collect_health(genuine)

    # ---- 요약문 생성
    parts = []
    n_voc = len(genuine)
    parts.append(f"발성 {len(segs)}개 중 유효 발성 {n_voc}개" +
                 (f", 환경음/오검출 추정 {len(noise)}개" if noise else "") + ".")
    if ctx:
        parts.append(f"상황('{ctx}') 기준 대표 유형은 '{top_type}'.")
    else:
        parts.append(f"상황 메모 없음 — 음향 특징 기준 대표 유형은 '{top_type}'(추정).")
    if clean_meows:
        parts.append(f"구간 {','.join('#'+str(i) for i in clean_meows)}은 유성·HNR 양호로 맑은 발성.")
    if f0_solicit:
        lo, hi = int(min(f0_solicit)), int(max(f0_solicit))
        parts.append(f"요구형 F0 {lo}~{hi}Hz.")
    if rough:
        parts.append(f"구간 {','.join('#'+str(i) for i in rough)}에서 쉰 목소리 신호(HNR 낮음) — "
                     f"단일 클립 진단 아님, 반복 시 다음 정기 검진에서 수의사 언급 권장.")
    if noise:
        parts.append("무성·초단발 구간은 움직임/환경음 가능성이 커 대표 유형에서 제외함.")

    summary = " ".join(parts)

    data["top_type"] = top_type
    data["summary"] = summary
    data["interpretation"] = {
        "engine": "rule-based-v1",
        "context_label": ctx_label,
        "genuine_segments": [s["segment"] for s in genuine],
        "noise_segments": [s["segment"] for s in noise],
        "clean_segments": clean_meows,
        "rough_segments": rough,
        "hoarse_count": hoarse,
        "type_counts": type_counts,
    }
    return data


def main():
    if len(sys.argv) < 2:
        print("usage: python interpret.py analysis.json", file=sys.stderr); sys.exit(1)
    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data = interpret(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[interpret] top_type={data['top_type']}")
    print(f"[interpret] summary={data['summary']}")


if __name__ == "__main__":
    main()
