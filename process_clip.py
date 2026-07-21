#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_clip.py — 클립 자동 처리 오케스트레이터
------------------------------------------------
GitHub Actions 에서 실행된다.

동작:
  1) 사이트에서 클립 메타/파일 다운로드
  2) meow_analyze.py 로 음향 지표 + 스펙트로그램 생성
  3) interpret.py 로 규칙기반 top_type/summary 자동 생성
  4) 결과(JSON) + 스펙트로그램(PNG) 을 사이트로 POST, analyzed=true 표시

모드:
  --clip-id XXXX   지정한 클립 하나 처리 (repository_dispatch / 수동 실행)
  --all-pending    아직 분석 안 된(analyzed=false) 클립 전부 처리 (스케줄 폴링)

환경변수:
  SITE_BASE   사이트 주소 (기본값 아래 BASE)
  CAT_NAME    고양이 이름 (기본 '우리집 고양이')
"""
import os, sys, json, subprocess, tempfile, argparse, urllib.request, urllib.error

BASE = os.environ.get("SITE_BASE", "https://cat-meow.sungsangkyung77.workers.dev").rstrip("/")
CAT  = os.environ.get("CAT_NAME", "우리집 고양이")
HERE = os.path.dirname(os.path.abspath(__file__))
HISTORY = os.path.join(HERE, "history.json")   # 누적 기록(개체 기준선)


def http_get(url, timeout=60):
    req = urllib.request.Request(url, headers={"user-agent": "catmeow-bot"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def http_post(url, data, content_type, timeout=120):
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"content-type": content_type,
                                          "user-agent": "catmeow-bot"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def list_clips():
    return json.loads(http_get(f"{BASE}/api/clips").decode("utf-8"))["items"]


def process_one(clip):
    cid = clip["id"]
    ext = clip.get("ext", "mov")
    ctx = clip.get("context", "") or ""
    print(f"\n=== 처리 시작: {cid} (상황='{ctx or '없음'}') ===")

    with tempfile.TemporaryDirectory() as td:
        clip_path = os.path.join(td, f"clip.{ext}")
        # 1) 다운로드
        data = http_get(f"{BASE}/media/clips/{cid}.{ext}", timeout=180)
        with open(clip_path, "wb") as f:
            f.write(data)
        print(f"[download] {len(data)} bytes")

        # 2) 분석
        out = os.path.join(td, "out")
        os.makedirs(out, exist_ok=True)
        cmd = [sys.executable, os.path.join(HERE, "meow_analyze.py"), clip_path,
               "--out", out, "--cat", CAT, "--context", ctx]
        if os.path.exists(HISTORY):
            cmd += ["--history", HISTORY]
        else:
            cmd += ["--history", os.path.join(out, "history.json")]
        subprocess.run(cmd, check=True)

        # 산출물 경로 찾기
        aj = next((os.path.join(out, x) for x in os.listdir(out) if x.endswith("_analysis.json")), None)
        png = next((os.path.join(out, x) for x in os.listdir(out) if x.endswith("_spectrogram.png")), None)
        hist = os.path.join(out, "history.json")
        if not aj or not png:
            raise RuntimeError("분석 산출물 없음")

        # 3) 규칙기반 해석
        subprocess.run([sys.executable, os.path.join(HERE, "interpret.py"), aj], check=True)

        # 상황 메모를 분석 JSON 에도 반영
        with open(aj, encoding="utf-8") as f:
            adata = json.load(f)
        adata["context_note"] = ctx
        with open(aj, "w", encoding="utf-8") as f:
            json.dump(adata, f, ensure_ascii=False)

        # 4) 게시
        with open(aj, "rb") as f:
            r1 = http_post(f"{BASE}/api/analysis?id={cid}", f.read(), "application/json")
        with open(png, "rb") as f:
            r2 = http_post(f"{BASE}/api/spectrogram?id={cid}", f.read(), "image/png")
        print(f"[post] analysis={r1} spectrogram={r2}")
        print(f"[done] {cid} → top_type={adata.get('top_type')}")

        # 누적 기록 갱신(다음 실행에서 개체 기준선으로 사용)
        if os.path.exists(hist):
            try:
                import shutil; shutil.copy(hist, HISTORY)
            except Exception as e:
                print(f"[warn] history 갱신 실패: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-id")
    ap.add_argument("--all-pending", action="store_true")
    args = ap.parse_args()

    clips = list_clips()
    if args.clip_id:
        target = [c for c in clips if c["id"] == args.clip_id]
        if not target:
            print(f"[error] 클립 없음: {args.clip_id}"); sys.exit(1)
    elif args.all_pending:
        target = [c for c in clips if not c.get("analyzed")]
        print(f"[poll] 미분석 클립 {len(target)}개")
    else:
        print("사용법: --clip-id XXXX  또는  --all-pending"); sys.exit(1)

    errors = 0
    for c in target:
        try:
            process_one(c)
        except Exception as e:
            errors += 1
            print(f"[error] {c['id']} 처리 실패: {e}", file=sys.stderr)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
