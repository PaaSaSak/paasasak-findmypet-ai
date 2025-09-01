#!/usr/bin/env python3
"""
Oxford-IIIT Pet → class-per-id 폴더 구조 변환 스크립트
- 입력: dataset/oxford-pet/
    ├─ images/                    # 원본 이미지 (*.jpg)
    └─ annotations/
        ├─ trainval.txt           # 학습 리스트
        ├─ test.txt               # 테스트 리스트 (여기서는 val로 사용)
        ├─ xmls/                  # 바운딩박스/품종 정보 (이번 스크립트는 사용 안 함)
        └─ trimaps/               # 세그멘테이션 (사용 안 함)

- 출력: dataset/embed/
    ├─ train/<class_name>/*.jpg
    └─ val/<class_name>/*.jpg

* class_name 은 기본적으로 파일 basename의 prefix (예: 'Abyssinian_123' → 'Abyssinian').
* species 필터 지원: all | cat | dog (annotations의 3번째 필드 1=cat, 2=dog)
* copy / link 모드 지원: 기본 copy, 같은 디스크/파티션이면 하드링크(link) 사용 가능
"""

import argparse
import os
import shutil
from pathlib import Path


def parse_list_file(list_path: Path):
    """
    annotations/{trainval.txt, test.txt} 를 파싱하여
    [(basename, class_id, species, breed_id), ...] 리스트를 반환.

    파일 포맷(공식 문서 기준):
      <basename> <class_id> <species> <breed_id>
        - basename: 파일명에서 확장자(.jpg) 뺀 이름
        - class_id: 1..37(cat) 또는 1..25(dog) (통합 표기)
        - species : 1=cat, 2=dog
        - breed_id: 품종별 id
    """
    items = []
    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            basename = parts[0]
            class_id = int(parts[1])
            species = int(parts[2])  # 1=cat, 2=dog
            breed_id = int(parts[3])
            items.append((basename, class_id, species, breed_id))
    return items


def want_species(spec_filter: str, species_code: int) -> bool:
    """species 필터링 (all | cat | dog)"""
    if spec_filter == "all":
        return True
    if spec_filter == "cat" and species_code == 1:
        return True
    if spec_filter == "dog" and species_code == 2:
        return True
    return False


def safe_class_name(name: str) -> str:
    """
    클래스 폴더명 안전화:
      - 앞뒤 공백 제거
      - 공백을 '_'로 치환
      - 파일 시스템에 문제될 수 있는 문자 최소 치환
    """
    name = name.strip()
    name = name.replace(" ", "_")
    name = name.replace("/", "_").replace("\\", "_").replace(":", "_")
    return name


def resolve_image_path(images_dir: Path, basename: str, lowercase_fallback: bool) -> Path | None:
    """
    basename에서 확장자만 붙여 실제 파일 경로를 찾는다.
    - 기본: .jpg
    - lowercase_fallback=True면 .JPG, .jpeg 등도 시도
    """
    cand = images_dir / f"{basename}.jpg"
    if cand.exists():
        return cand
    if lowercase_fallback:
        for ext in (".JPG", ".jpeg", ".JPEG", ".png", ".PNG"):
            cand2 = images_dir / f"{basename}{ext}"
            if cand2.exists():
                return cand2
    return None


def place_split(
    split_items,
    images_dir: Path,
    out_root: Path,
    species_filter: str,
    mode: str,
    lowercase: bool,
    max_per_class: int | None = None,
):
    """
    한 split(train/val) 리스트를 순회하며 out_root/<class_name>/로 복사(또는 하드링크)한다.
    - class_name: basename의 '_' 앞 prefix (예: 'Abyssinian_123' → 'Abyssinian')
    - species_filter: cat/dog/all
    - mode: 'copy' | 'link'
    - lowercase: True면 확장자 대체 탐색 허용
    - max_per_class: 클래스당 최대 파일 수 제한 (디버그/샘플링용)
    """
    placed, skipped = 0, 0
    per_class_count: dict[str, int] = {}

    for basename, class_id, species, breed_id in split_items:
        if not want_species(species_filter, species):
            skipped += 1
            continue

        # 클래스 이름 = 파일 prefix (품종명)
        cls = basename.rsplit("_", 1)[0]
        cls = safe_class_name(cls)
        cls_dir = out_root / cls
        cls_dir.mkdir(parents=True, exist_ok=True)

        # per-class 제한이 있으면 체크
        if max_per_class is not None:
            if per_class_count.get(cls, 0) >= max_per_class:
                skipped += 1
                continue

        # 원본 이미지 경로 해석
        src_jpg = resolve_image_path(images_dir, basename, lowercase_fallback=lowercase)
        if src_jpg is None or not src_jpg.exists():
            skipped += 1
            continue

        dst_jpg = cls_dir / f"{basename}.jpg"

        # copy vs link
        if mode == "link":
            try:
                # 하드링크 생성 (같은 볼륨/권한 필요). 실패 시 복사로 대체
                os.link(src_jpg, dst_jpg)
            except Exception:
                shutil.copy2(src_jpg, dst_jpg)
        else:
            shutil.copy2(src_jpg, dst_jpg)

        per_class_count[cls] = per_class_count.get(cls, 0) + 1
        placed += 1

    return placed, skipped


def main():
    ap = argparse.ArgumentParser(
        description="Convert Oxford-IIIT Pet to class-per-id folder structure (breed-as-id)."
    )
    ap.add_argument("--src", default="dataset/oxford-pet", help="Oxford root (contains images/ and annotations/)")
    ap.add_argument("--dst", default="dataset/embed", help="Output root for embed/{train,val}/<class>/*.jpg")
    ap.add_argument("--species", choices=["all", "cat", "dog"], default="all", help="Use only cat/dog/all")
    ap.add_argument("--mode", choices=["copy", "link"], default="copy", help="copy files or create hardlinks if possible")
    ap.add_argument("--lowercase", action="store_true", help="Try alternative extensions (.JPG/.jpeg) if not found")
    ap.add_argument("--max_per_class", type=int, default=None, help="Limit per-class images (debug/sampling)")
    args = ap.parse_args()

    src = Path(args.src)
    images_dir = src / "images"
    ann_dir = src / "annotations"

    train_list = ann_dir / "trainval.txt"
    val_list = ann_dir / "test.txt"  # 우리는 test를 val로 사용

    # 기본 파일 체크
    if not images_dir.exists():
        raise FileNotFoundError(f"Not found: {images_dir}")
    if not train_list.exists() or not val_list.exists():
        raise FileNotFoundError(f"Not found: {train_list} or {val_list}")

    # 출력 디렉토리
    dst_root = Path(args.dst)
    dst_train = dst_root / "train"
    dst_val = dst_root / "val"
    dst_train.mkdir(parents=True, exist_ok=True)
    dst_val.mkdir(parents=True, exist_ok=True)

    # 리스트 파싱
    train_items = parse_list_file(train_list)
    val_items = parse_list_file(val_list)

    # 변환 실행
    tr_placed, tr_skipped = place_split(
        train_items, images_dir, dst_train, args.species, args.mode, args.lowercase, args.max_per_class
    )
    va_placed, va_skipped = place_split(
        val_items, images_dir, dst_val, args.species, args.mode, args.lowercase, args.max_per_class
    )

    # 결과 요약
    print(f"[DONE] train: placed={tr_placed}, skipped={tr_skipped}")
    print(f"[DONE] val  : placed={va_placed}, skipped={va_skipped}")
    print(f"Output root: {dst_root.resolve()}")
    if args.max_per_class:
        print(f"(per-class limit: {args.max_per_class})")


if __name__ == "__main__":
    main()
