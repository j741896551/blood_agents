"""OCR 服务适配层；可选重依赖只在真正识别图片时导入。"""

from __future__ import annotations

from pathlib import Path


class OCRService:
    """PaddleOCR 优先、Tesseract 降级；依赖按需加载以保持基础服务轻量。"""

    def __init__(self, engine: str = "auto") -> None:
        """初始化 OCRService 的依赖与运行状态。"""
        self.engine = engine

    def extract(self, image_path: str) -> dict:
        """从指定图片提取 OCR 文本或返回不可用状态。"""
        path = Path(image_path)
        if not path.is_file():
            raise ValueError("图片文件不存在")
        if self.engine in {"auto", "paddle"}:
            try:
                from paddleocr import PaddleOCR

                rows = PaddleOCR(use_angle_cls=True, lang="ch").ocr(str(path), cls=True)
                text = "\n".join(line[1][0] for block in rows for line in (block or []))
                return {"engine": "paddleocr", "text": text}
            except ImportError:
                if self.engine == "paddle":
                    raise RuntimeError("请安装 paddleocr 与 paddlepaddle") from None
        try:
            import pytesseract
            from PIL import Image

            return {"engine": "tesseract", "text": pytesseract.image_to_string(Image.open(path), lang="chi_sim+eng")}
        except ImportError as exc:
            raise RuntimeError("OCR 未安装：请安装 PaddleOCR 或 pytesseract + Pillow") from exc

