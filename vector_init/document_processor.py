from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


SUPPORTED_SUFFIXES = {
    ".pdf",
    ".txt",
    ".md",
}


def parse_pdf(path: Path) -> list[Document]:
    documents = []

    try:
        reader = PdfReader(
            str(path),
            strict=False,
        )
    except Exception as exc:
        print(f"[失败] 无法打开 PDF：{path}")
        print(f"       {type(exc).__name__}: {exc}")
        return documents

    print(f"[PDF] {path}，共 {len(reader.pages)} 页")

    for page_number, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            print(
                f"[跳过] {path} 第 {page_number + 1} 页"
                f"解析失败：{type(exc).__name__}: {exc}"
            )
            continue

        document = Document(
            page_content=text,
            metadata={
                "source": path.as_posix(),
                "page": page_number,
                "file_type": "pdf",
            },
        )

        documents.append(document)

    return documents


def parse_text_file(path: Path) -> list[Document]:
    encodings = [
        "utf-8",
        "utf-8-sig",
        "gb18030",
    ]

    text = None
    last_error = None

    for encoding in encodings:
        try:
            text = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
        except Exception as exc:
            print(f"[失败] 无法读取文件 {path}: {exc}")
            return []

    if text is None:
        print(f"[失败] 无法识别文件编码：{path}")
        print(f"       {last_error}")
        return []

    document = Document(
        page_content=text,
        metadata={
            "source": path.as_posix(),
            "file_type": path.suffix.lower().lstrip("."),
        },
    )

    return [document]


def process_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return parse_pdf(path)

    if suffix in {".txt", ".md"}:
        return parse_text_file(path)

    print(f"[跳过] 不支持的文件：{path}")
    return []


def process_directory(
    data_dir: Path,
) -> list[Document]:
    if not data_dir.exists():
        raise RuntimeError(
            f"数据目录不存在：{data_dir.resolve()}"
        )

    documents = []

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            print(f"[跳过] 不支持的文件：{path}")
            continue

        parsed_documents = process_file(path)
        documents.extend(parsed_documents)

    return documents
