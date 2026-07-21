#!/usr/bin/env python3
"""
고양이 울음소리 분석기 (cat-meow-analyzer)

영상/오디오 파일에서 울음 구간을 자동 검출하고, 구간별 음향 지표를 산출한 뒤
스펙트로그램 이미지와 JSON 리포트를 생성한다.

사용법:
    python3 meow_analyze.py <영상|오디오 파일> [--out 출력폴더] [--cat 고양이이름]
                            [--context "상황 메모"] [--history history.json]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime

import numpy as np
from scipy.io import wavfile
from scipy import signal as sps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 한글 라벨용 CJK 폰트 등록 (없으면 조용히 기본 폰트 사용)
for _p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
           "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
    if os.path.exists(_p):
        try:
            font_manager.fontManager.addfont(_p)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=_p).get_name()
            break
        except Exception:
            pass
plt.rcParams["axes.unicode_minus"] = False

SR = 22050
FRAME = 1024
HOP = 256
F0_MIN, F0_MAX = 55.0, 1800.0


# ---------------------------------------------------------------- 오디오 로드
def load_audio(path):
    """ffmpeg로 어떤 포맷이든 mono 22050Hz float 배열로 변환."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav_path = tf.name
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", path,
           "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", wav_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg 오디오 추출 실패: {r.stderr.strip()[:400]}")
    sr, x = wavfile.read(wav_path)
    os.unlink(wav_path)
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    peak = np.abs(x).max()
    if peak > 0:
        x = x / peak
    return sr, x


# ------------------------------------------------------------------ 프레이밍
def frames_of(x, frame=FRAME, hop=HOP):
    n = 1 + max(0, (len(x) - frame) // hop)
    if n <= 0 or len(x) < frame:
        return np.zeros((0, frame))
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def frame_times(n, hop=HOP, frame=FRAME, sr=SR):
    return (np.arange(n) * hop + frame / 2) / sr


# ------------------------------------------------------- 울음 구간 자동 검출
def detect_segments(x, sr=SR, min_dur=0.12, merge_gap=0.18):
    """단시간 에너지 기반 적응형 임계값으로 발성 구간을 잘라낸다."""
    fr = frames_of(x)
    if len(fr) == 0:
        return []
    rms = np.sqrt((fr ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)

    noise_floor = np.percentile(db, 20)
    peak_db = db.max()
    thr = max(noise_floor + 9.0, peak_db - 22.0)
    if peak_db - noise_floor < 6.0:
        return []

    active = db > thr
    segs = []
    i = 0
    while i < len(active):
        if active[i]:
            j = i
            while j + 1 < len(active) and active[j + 1]:
                j += 1
            segs.append([i, j])
            i = j + 1
        else:
            i += 1

    merged = []
    gap_frames = int(merge_gap * sr / HOP)
    for s in segs:
        if merged and s[0] - merged[-1][1] <= gap_frames:
            merged[-1][1] = s[1]
        else:
            merged.append(s)

    out = []
    min_frames = int(min_dur * sr / HOP)
    for s, e in merged:
        if e - s + 1 < min_frames:
            continue
        a = max(0, s * HOP)
        b = min(len(x), e * HOP + FRAME)
        out.append((a, b))
    return out


# --------------------------------------------------------------- F0 / 음향지표
def f0_track(seg, sr=SR):
    """정규화 자기상관 기반 F0 추정 + 유성음 신뢰도(HNR 근사)."""
    fr = frames_of(seg, FRAME, HOP)
    f0s, confs = [], []
    win = np.hanning(FRAME)
    lag_min = int(sr / F0_MAX)
    lag_max = min(int(sr / F0_MIN), FRAME - 1)
    for f in fr:
        f = (f - f.mean()) * win
        e = np.dot(f, f)
        if e < 1e-9:
            f0s.append(np.nan); confs.append(0.0); continue
        ac = np.correlate(f, f, mode="full")[FRAME - 1:]
        ac = ac / (ac[0] + 1e-12)
        window = ac[lag_min:lag_max]
        if len(window) < 3:
            f0s.append(np.nan); confs.append(0.0); continue
        k = int(np.argmax(window)) + lag_min
        if 0 < k < len(ac) - 1:
            a, b, c = ac[k - 1], ac[k], ac[k + 1]
            denom = (a - 2 * b + c)
            shift = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
        else:
            shift = 0.0
        r = float(ac[k])
        f0 = sr / (k + shift)
        if r < 0.30 or not (F0_MIN <= f0 <= F0_MAX):
            f0s.append(np.nan)
        else:
            f0s.append(f0)
        confs.append(max(0.0, min(0.999, r)))
    return np.array(f0s), np.array(confs) if confs else np.array([0.0])


def contour_shape(f0):
    """피치 곡선 형태: rising / falling / arch / dip / flat / modulated."""
    v = f0[~np.isnan(f0)]
    if len(v) < 4:
        return "unknown", 0.0
    n = len(v)
    lv = np.log2(v)
    total = (lv[-1] - lv[0]) * 12
    third = max(1, n // 3)
    a, b, c = lv[:third].mean(), lv[third:2 * third].mean(), lv[2 * third:].mean()
    k = min(5, max(3, (len(lv) // 2) * 2 + 1))
    if k % 2 == 0:
        k += 1
    sm = sps.medfilt(lv, kernel_size=min(k, len(lv) if len(lv) % 2 else len(lv) - 1))
    d = np.diff(sm)
    d = d[np.abs(d) > 0.01]
    turns = int(np.sum(np.diff(np.sign(d)) != 0)) if len(d) > 2 else 0

    if turns >= 5:
        return "modulated", float(turns)
    if b > a + 0.08 and b > c + 0.08:
        return "arch", float(total)
    if b < a - 0.08 and b < c - 0.08:
        return "dip", float(total)
    if total > 1.5:
        return "rising", float(total)
    if total < -1.5:
        return "falling", float(total)
    return "flat", float(total)


def low_freq_purr_check(seg, sr=SR):
    """그르렁(purr): 20~60Hz 대역 에너지 비율 (연구상 25~50Hz 우세)."""
    f, P = sps.welch(seg, fs=sr, nperseg=min(4096, max(256, len(seg))))
    total = P.sum() + 1e-15
    return float(P[(f >= 20) & (f <= 60)].sum() / total)


def analyze_segment(x, a, b, sr=SR):
    seg = x[a:b]
    dur = (b - a) / sr
    f0, conf = f0_track(seg, sr)
    v = f0[~np.isnan(f0)]
    voiced_ratio = float(len(v) / max(1, len(f0)))

    f, P = sps.welch(seg, fs=sr, nperseg=min(2048, max(256, len(seg))))
    Pn = P / (P.sum() + 1e-15)
    centroid = float((f * Pn).sum())
    cumsum = np.cumsum(Pn)
    rolloff = float(f[min(np.searchsorted(cumsum, 0.85), len(f) - 1)])
    spread = float(np.sqrt(((f - centroid) ** 2 * Pn).sum()))
    flatness = float(np.exp(np.log(P + 1e-15).mean()) / (P.mean() + 1e-15))

    rms = float(np.sqrt((seg ** 2).mean()))
    db = float(20 * np.log10(rms + 1e-12))
    cm = float(conf.mean()) if len(conf) else 0.0
    hnr = float(10 * np.log10(max(cm, 1e-3) / max(1 - cm, 1e-3)))
    shape, shape_val = contour_shape(f0)

    env = np.abs(sps.hilbert(seg)) if len(seg) < 400000 else np.abs(seg)
    if len(env) > 101:
        env = sps.medfilt(env, kernel_size=101)
    pk = env.max() + 1e-12
    try:
        i10 = int(np.argmax(env > 0.1 * pk))
        i90 = int(np.argmax(env > 0.9 * pk))
        attack = max(0.0, (i90 - i10) / sr)
    except Exception:
        attack = float("nan")

    return {
        "start_s": round(a / sr, 3),
        "end_s": round(b / sr, 3),
        "duration_s": round(dur, 3),
        "f0_mean_hz": round(float(np.nanmean(v)), 1) if len(v) else None,
        "f0_min_hz": round(float(np.nanmin(v)), 1) if len(v) else None,
        "f0_max_hz": round(float(np.nanmax(v)), 1) if len(v) else None,
        "f0_range_semitones": round(float(12 * np.log2(v.max() / v.min())), 2) if len(v) > 1 else None,
        "f0_start_hz": round(float(v[0]), 1) if len(v) else None,
        "f0_end_hz": round(float(v[-1]), 1) if len(v) else None,
        "contour": shape,
        "contour_value": round(shape_val, 2),
        "voiced_ratio": round(voiced_ratio, 3),
        "hnr_db": round(hnr, 2),
        "spectral_centroid_hz": round(centroid, 1),
        "spectral_rolloff85_hz": round(rolloff, 1),
        "spectral_spread_hz": round(spread, 1),
        "spectral_flatness": round(flatness, 4),
        "level_db": round(db, 2),
        "attack_s": round(attack, 4) if attack == attack else None,
        "purr_band_ratio": round(low_freq_purr_check(seg, sr), 4),
        "_f0_track": [None if np.isnan(z) else round(float(z), 1) for z in f0],
    }


# ------------------------------------------------------------- 규칙기반 분류
def classify(m):
    """음향 지표 → 발성 유형 후보(점수순). 최종 해석은 맥락과 함께 판단."""
    cands = []

    def add(name, score, why):
        cands.append({"type": name, "score": round(min(1.0, score), 2), "evidence": why})

    dur = m["duration_s"]
    f0 = m["f0_mean_hz"] or 0
    rng = m["f0_range_semitones"] or 0
    flat = m["spectral_flatness"]
    vr = m["voiced_ratio"]
    cont = m["contour"]
    cen = m["spectral_centroid_hz"]

    if m["purr_band_ratio"] > 0.18 and vr < 0.45:
        add("purr(그르렁)", 0.55 + m["purr_band_ratio"],
            "20~60Hz 저역 에너지 우세 + 유성 비율 낮음")

    # 하악/쉭 — 유성 성분이 거의 없고 고역 잡음이 우세한 경우
    if vr < 0.30 and cen > 2000 and m["hnr_db"] < 0:
        sc = 0.55 + min(0.3, flat * 1.5)
        add("hiss/spit(하악)", sc,
            f"무성 잡음(유성비 {vr}) + 고역중심 {cen}Hz + HNR {m['hnr_db']}dB")

    if f0 and f0 < 250 and dur > 0.6 and m["hnr_db"] < 6:
        add("growl(위협 그르릉)", 0.6,
            f"저주파 F0 {f0}Hz + 지속 {dur}s + 낮은 HNR")

    if dur < 0.55 and cont in ("rising", "arch") and f0 > 380:
        add("trill/chirrup(트릴·인사)", 0.65,
            f"짧은 길이 {dur}s + {cont} 곡선 + 높은 F0 {f0}Hz")

    if 0.2 <= dur <= 1.6 and 221 <= f0 <= 1185 and vr > 0.45:
        base = 0.62
        if 0.3 <= dur <= 0.7:              # 연구상 평균 길이 0.42s
            base += 0.06
        if dur > 1.0:                       # 길게 끌면 맥락형 쪽으로 무게 이동
            base -= 0.15
        add("meow(일반 야옹)", base, f"길이 {dur}s + F0 {f0}Hz + {cont} 곡선")

    # Schötz(Meowsic): 먹이·요구 상황의 야옹은 상승 억양, 스트레스·불편은 하강 억양
    if cont in ("rising", "arch") and 221 <= f0 <= 1185 and vr > 0.45 and dur >= 0.25:
        sc = 0.62 + (0.08 if cont == "rising" else 0.0) + min(0.1, max(0.0, (dur - 0.6) * 0.15))
        add("food/solicitation meow(먹이·요구)", sc,
            f"{cont} 억양 + F0 {f0}Hz — 먹이·요구 맥락에서 관찰되는 상승 억양형")

    if cont in ("falling", "flat") and vr > 0.45 and 221 <= f0 <= 1185 and dur >= 0.4:
        sc = 0.6 + min(0.15, max(0.0, (dur - 0.8) * 0.2))
        add("stress/discomfort meow(스트레스·불편)", sc,
            f"{cont} 억양 + 길이 {dur}s — 진료·이동 등 불편 맥락에서 관찰되는 하강 억양형")

    if cont == "modulated" and 0.4 <= dur <= 2.0:
        add("complaint(불만·항의)", 0.55, "피치 곡선 변조가 잦음")

    if f0 > 950 or (m["f0_max_hz"] or 0) > 1300:
        sc = 0.5 + (0.2 if (m["attack_s"] is not None and m["attack_s"] < 0.05) else 0.0)
        add("pain/distress(통증·놀람)", sc,
            f"고주파 F0 {f0}Hz / 최고 {m['f0_max_hz']}Hz — 정상 야옹 상한(약 1185Hz)을 넘음")

    if dur > 1.5 and rng > 7:
        add("yowl/caterwaul(길게 우는 소리)", 0.6,
            f"긴 지속 {dur}s + 넓은 음역 {rng}반음 — 발정·불안·노령 인지장애 감별 필요")

    if not cands:
        add("unclassified(분류 보류)", 0.2, "기준표의 어느 유형과도 뚜렷이 맞지 않음")

    cands.sort(key=lambda c: -c["score"])
    return cands


def health_flags(m):
    """건강 이상 시사 신호. 진단이 아니라 수의사 상담 참고용."""
    flags = []
    if m["hnr_db"] < 2 and (m["voiced_ratio"] or 0) > 0.4:
        flags.append({"flag": "쉰 목소리(hoarseness) 가능",
                      "detail": f"유성 구간인데 HNR {m['hnr_db']}dB로 매우 거칢 — 후두염·성대 이상·상부호흡기 감염에서 나타남"})
    if m["spectral_flatness"] > 0.25 and m["spectral_centroid_hz"] > 3000 and (m["voiced_ratio"] or 0) < 0.3:
        flags.append({"flag": "천명음/재채기성 잡음 가능",
                      "detail": "광대역 잡음 성분 우세 — 하악질이 아니라면 호흡기 잡음 여부 확인 필요"})
    if m["duration_s"] > 2.0 and (m["f0_range_semitones"] or 0) > 9:
        flags.append({"flag": "장시간 고음 울음",
                      "detail": "노령묘라면 갑상선기능항진증·고혈압·인지기능장애, 미중성화묘라면 발정 감별"})
    if (m["f0_mean_hz"] or 0) > 950 and (m["attack_s"] or 1) < 0.04:
        flags.append({"flag": "급성 통증 반응 가능",
                      "detail": "매우 높은 F0 + 순간적 어택 — 직전 신체 접촉·낙상 여부 확인"})
    if (m["f0_mean_hz"] or 999) < 180 and m["duration_s"] > 1.0 and m["hnr_db"] < 4:
        flags.append({"flag": "이상 저음·거친 발성",
                      "detail": "지속적이면 후두·기도 협착 감별 필요"})
    return flags


# ------------------------------------------------------------------ 시각화
def hz2mel(f):
    return 2595 * np.log10(1 + np.asarray(f, dtype=float) / 700)


def mel_filters(n_mels, n_fft, sr, fmin=50, fmax=None):
    fmax = fmax or sr / 2
    def mel2hz(m): return 700 * (10 ** (m / 2595) - 1)
    m_pts = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    f_pts = mel2hz(m_pts)
    bins = np.floor((n_fft + 1) * f_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        r = min(r, fb.shape[1] - 1)
        c = min(max(c, l + 1), r - 1)
        if l < 0 or c <= l or r <= c:
            continue
        fb[i, l:c] = np.linspace(0, 1, c - l)
        fb[i, c:r] = np.linspace(1, 0, r - c)
    return fb


def plot_overview(x, sr, segs, seg_metrics, out_png, title):
    n_fft = 1024
    f, t, Z = sps.stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - HOP, window="hann")
    S = np.abs(Z)
    fmax = min(8000, sr / 2)
    fb = mel_filters(96, n_fft, sr, fmin=50, fmax=fmax)
    M = fb @ (S ** 2)
    Mdb = 10 * np.log10(M + 1e-10)
    Mdb = np.maximum(Mdb, Mdb.max() - 65)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#11131a")

    ax = axes[0]
    ax.imshow(Mdb, origin="lower", aspect="auto", cmap="magma",
              extent=[t[0], t[-1], 0, Mdb.shape[0]])
    ticks_hz = [100, 250, 500, 1000, 2000, 4000, 8000]
    mlo, mhi = hz2mel(50), hz2mel(fmax)
    pos, lab = [], []
    for h in ticks_hz:
        if h <= fmax:
            pos.append((hz2mel(h) - mlo) / (mhi - mlo) * Mdb.shape[0])
            lab.append(str(h))
    ax.set_yticks(pos)
    ax.set_yticklabels(lab)
    ax.set_ylabel("Frequency (Hz)", color="#e8e8ea")

    for i, ((a, b), m) in enumerate(zip(segs, seg_metrics), 1):
        ax.axvspan(a / sr, b / sr, color="#4cc9f0", alpha=0.13)
        ax.text(a / sr, Mdb.shape[0] * 0.94, f"#{i}", color="#4cc9f0",
                fontsize=11, fontweight="bold")
        tr = np.array([np.nan if z is None else z for z in m["_f0_track"]], dtype=float)
        if len(tr):
            tt = a / sr + frame_times(len(tr))
            mel_y = (hz2mel(np.clip(tr, 50, fmax)) - mlo) / (mhi - mlo) * Mdb.shape[0]
            ax.plot(tt, mel_y, color="#7CFFB2", lw=1.6)

    ax2 = axes[1]
    tx = np.arange(len(x)) / sr
    ax2.plot(tx, x, color="#8d99ae", lw=0.4)
    for a, b in segs:
        ax2.axvspan(a / sr, b / sr, color="#4cc9f0", alpha=0.18)
    ax2.set_xlabel("Time (s)", color="#e8e8ea")
    ax2.set_ylabel("Waveform", color="#e8e8ea")

    for a in axes:
        a.set_facecolor("#11131a")
        a.tick_params(colors="#b8bcc8")
        for s in a.spines.values():
            s.set_color("#2a2e3a")
    axes[0].set_title(title, color="#e8e8ea", fontsize=13, pad=10)
    plt.tight_layout()
    plt.savefig(out_png, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", default=".")
    ap.add_argument("--cat", default="cat")
    ap.add_argument("--context", default="")
    ap.add_argument("--history", default="")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input))[0]

    sr, x = load_audio(args.input)
    dur_total = len(x) / sr
    segs = detect_segments(x, sr)
    seg_metrics = [analyze_segment(x, a, b, sr) for a, b in segs]

    png = os.path.join(args.out, f"{stem}_spectrogram.png")
    if len(x) > FRAME:
        plot_overview(x, sr, segs, seg_metrics, png,
                      f"{args.cat} - {stem} ({dur_total:.1f}s, segments: {len(segs)})")

    results = []
    for i, m in enumerate(seg_metrics, 1):
        pub = {k: v for k, v in m.items() if not k.startswith("_")}
        results.append({
            "segment": i,
            "metrics": pub,
            "candidates": classify(m),
            "health_flags": health_flags(m),
        })

    report = {
        "file": os.path.basename(args.input),
        "cat": args.cat,
        "analyzed_at": datetime.now().isoformat(timespec="seconds"),
        "context_note": args.context,
        "audio_duration_s": round(dur_total, 2),
        "sample_rate": sr,
        "segments_detected": len(segs),
        "spectrogram": os.path.basename(png),
        "segments": results,
    }

    if args.history:
        hist = []
        if os.path.exists(args.history):
            try:
                hist = json.load(open(args.history, encoding="utf-8"))
            except Exception:
                hist = []
        hist.append({
            "analyzed_at": report["analyzed_at"],
            "file": report["file"],
            "cat": report["cat"],
            "context_note": report["context_note"],
            "segments": [{
                "duration_s": s["metrics"]["duration_s"],
                "f0_mean_hz": s["metrics"]["f0_mean_hz"],
                "contour": s["metrics"]["contour"],
                "top_type": s["candidates"][0]["type"] if s["candidates"] else None,
                "flags": [fl["flag"] for fl in s["health_flags"]],
            } for s in results],
        })
        json.dump(hist, open(args.history, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        report["history_file"] = args.history
        report["history_entries"] = len(hist)

    out_json = os.path.join(args.out, f"{stem}_analysis.json")
    json.dump(report, open(out_json, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[saved] {out_json}\n[saved] {png}", file=sys.stderr)


if __name__ == "__main__":
    main()
