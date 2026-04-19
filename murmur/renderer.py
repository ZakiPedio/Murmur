"""Renderer: convert various input types into grayscale numpy arrays for the encoder.

Each renderer returns a 2D float64 array of shape (height, width) with values in [0.0, 1.0].
This shape corresponds to (freq_bins, time_bins) for the encoder.
"""

from __future__ import annotations

import ast
import logging
import math
import platform
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MONOSPACE_PATHS = {
    "Linux": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    ],
    "Darwin": [
        "/System/Library/Fonts/Menlo.ttc",
        "/Library/Fonts/Courier New.ttf",
    ],
    "Windows": [
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/cour.ttf",
    ],
}

_SYSTEM_FONT_PATHS = {
    "Linux": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ],
    "Darwin": [
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ],
    "Windows": [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/verdana.ttf",
    ],
}


def _find_monospace_font(size: int) -> "ImageFont.FreeTypeFont | ImageFont.ImageFont":
    """Find a monospace font on the current platform.

    Args:
        size: Font size in points.

    Returns:
        A Pillow font object, falling back to the default bitmap font if no
        TrueType monospace font is found.
    """
    system = platform.system()
    candidates = _MONOSPACE_PATHS.get(system, [])
    for path in candidates:
        if Path(path).exists():
            try:
                font = ImageFont.truetype(path, size)
                logger.debug("Loaded monospace font: %s", path)
                return font
            except Exception:
                continue
    logger.debug("No monospace TrueType font found; using default bitmap font.")
    return ImageFont.load_default()


def _find_system_font(size: int) -> "ImageFont.FreeTypeFont | ImageFont.ImageFont":
    """Find a general-purpose system font on the current platform.

    Args:
        size: Font size in points.

    Returns:
        A Pillow font object, falling back to the default bitmap font if none found.
    """
    system = platform.system()
    candidates = _SYSTEM_FONT_PATHS.get(system, [])
    for path in candidates:
        if Path(path).exists():
            try:
                font = ImageFont.truetype(path, size)
                logger.debug("Loaded system font: %s", path)
                return font
            except Exception:
                continue
    logger.debug("No system TrueType font found; using default bitmap font.")
    return ImageFont.load_default()


def _pil_to_grayscale_array(img: Image.Image, invert: bool = False) -> np.ndarray:
    """Convert a PIL image to a float64 grayscale array in [0, 1].

    Args:
        img: Any PIL image.
        invert: If True, invert the values (1.0 - result).

    Returns:
        2D float64 array with values in [0.0, 1.0].
    """
    gray = img.convert("L")
    arr = np.array(gray, dtype=np.float64) / 255.0
    if invert:
        arr = 1.0 - arr
    return arr


# ---------------------------------------------------------------------------
# 1. Image renderer
# ---------------------------------------------------------------------------


def render_image(
    path: str,
    width: int,
    height: int,
    invert: bool = False,
) -> np.ndarray:
    """Load and render an image file as a grayscale numpy array.

    Supports PNG, JPG, BMP, GIF (first frame), and WEBP.

    Args:
        path: Path to the image file.
        width: Output width in pixels.
        height: Output height in pixels.
        invert: If True, invert pixel values (1.0 - result).

    Returns:
        2D float64 array of shape (height, width) with values in [0.0, 1.0].

    Raises:
        FileNotFoundError: If the image file does not exist.
        OSError: If the file cannot be opened as an image.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    logger.info("Rendering image: %s -> (%d x %d)", path, width, height)
    with Image.open(p) as img:
        # For animated GIF, use only the first frame
        img.seek(0) if hasattr(img, "seek") else None
        img_copy = img.convert("L")
        img_resized = img_copy.resize((width, height), Image.LANCZOS)
    arr = np.array(img_resized, dtype=np.float64) / 255.0
    if invert:
        arr = 1.0 - arr
    return arr


# ---------------------------------------------------------------------------
# 2. Text renderer
# ---------------------------------------------------------------------------


def render_text(
    text: str,
    width: int | None = None,
    height: int | None = None,
    font_path: str | None = None,
    font_size: int = 48,
    invert: bool = False,
) -> np.ndarray:
    """Render a text string as a grayscale numpy array using Pillow.

    Supports multi-line text (split on newlines). If width/height are given,
    the text is centered in an image of that size. Otherwise the image is
    auto-sized to fit all text with 10% padding on each side.

    Args:
        text: The text to render. Use '\\n' for line breaks.
        width: Optional output width in pixels.
        height: Optional output height in pixels.
        font_path: Optional path to a .ttf/.otf font file.
        font_size: Font size in points (default 48).
        invert: If True, render white text on black background instead of
            black text on white, and invert the result.

    Returns:
        2D float64 array with values in [0.0, 1.0].
    """
    # Load font
    if font_path is not None:
        font = ImageFont.truetype(font_path, font_size)
        logger.debug("Using provided font: %s", font_path)
    else:
        font = _find_system_font(font_size)

    bg_color = 255
    text_color = 0

    lines = text.split("\n")

    # Measure text extents using a temporary scratch image
    scratch = Image.new("L", (4096, 4096), bg_color)
    draw = ImageDraw.Draw(scratch)

    line_bboxes = []
    for line in lines:
        if line == "":
            # Empty line: measure a space to get line height
            bbox = draw.textbbox((0, 0), " ", font=font)
        else:
            bbox = draw.textbbox((0, 0), line, font=font)
        line_bboxes.append(bbox)

    # Compute total text block size
    line_height = max((b[3] - b[1]) for b in line_bboxes) if line_bboxes else font_size
    line_spacing = int(line_height * 0.2)
    total_text_h = len(lines) * line_height + (len(lines) - 1) * line_spacing
    max_text_w = max((b[2] - b[0]) for b in line_bboxes) if line_bboxes else 0

    # Always render at natural size first, then resize to the target canvas if
    # width/height are given.  This prevents text from being clipped when the
    # requested canvas is narrower than the rendered text.
    pad_x = max(int(max_text_w * 0.10), 10)
    pad_y = max(int(total_text_h * 0.10), 10)
    render_w = max_text_w + 2 * pad_x
    render_h = total_text_h + 2 * pad_y

    img = Image.new("L", (render_w, render_h), bg_color)
    draw = ImageDraw.Draw(img)

    # Center the block of text vertically and each line horizontally
    block_top = (render_h - total_text_h) // 2
    y = block_top
    for i, (line, bbox) in enumerate(zip(lines, line_bboxes)):
        line_w = bbox[2] - bbox[0]
        x = (render_w - line_w) // 2
        draw.text((x, y), line, fill=text_color, font=font)
        y += line_height + line_spacing

    # Fit into the requested canvas while preserving aspect ratio, then center.
    # This prevents both clipping (text wider than canvas) and distortion
    # (stretching to fill canvas height when text is naturally short).
    target_w = width if width is not None else render_w
    target_h = height if height is not None else render_h
    if (target_w, target_h) != (render_w, render_h):
        scale = min(target_w / render_w, target_h / render_h)
        fit_w = max(1, int(render_w * scale))
        fit_h = max(1, int(render_h * scale))
        img = img.resize((fit_w, fit_h), Image.LANCZOS)
        # Paste centered onto a white canvas of the exact target size
        canvas = Image.new("L", (target_w, target_h), bg_color)
        paste_x = (target_w - fit_w) // 2
        paste_y = (target_h - fit_h) // 2
        canvas.paste(img, (paste_x, paste_y))
        img = canvas

    arr = np.array(img, dtype=np.float64) / 255.0
    if invert:
        arr = 1.0 - arr
    logger.info("Rendered text image: (%d x %d)", target_w, target_h)
    return arr


# ---------------------------------------------------------------------------
# 3. ASCII art renderer
# ---------------------------------------------------------------------------


def render_ascii(
    text: str,
    width: int | None = None,
    height: int | None = None,
    font_size: int = 16,
    invert: bool = False,
) -> np.ndarray:
    """Render ASCII/monospace text as a grayscale numpy array.

    Uses a monospace font to preserve character grid alignment and exact
    line breaks. Auto-sizes the image to fit all text unless width/height
    are specified.

    Args:
        text: The ASCII art or monospace text to render. '\\n' splits lines.
        width: Optional output width in pixels.
        height: Optional output height in pixels.
        font_size: Font size in points (default 16).
        invert: If True, invert pixel values (white-on-black).

    Returns:
        2D float64 array with values in [0.0, 1.0].
    """
    font = _find_monospace_font(font_size)
    bg_color = 255
    text_color = 0

    lines = text.split("\n")

    scratch = Image.new("L", (8192, 4096), bg_color)
    draw = ImageDraw.Draw(scratch)

    # Measure a fixed character to determine cell size (monospace)
    ref_bbox = draw.textbbox((0, 0), "M", font=font)
    char_w = ref_bbox[2] - ref_bbox[0]
    char_h = ref_bbox[3] - ref_bbox[1]

    max_cols = max((len(line) for line in lines), default=0)
    num_rows = len(lines)

    pad = max(int(char_h * 0.5), 4)
    auto_w = max_cols * char_w + 2 * pad
    auto_h = num_rows * char_h + 2 * pad

    # Always render at natural size, then fit-within-box resize to the target
    # canvas. Prevents clipping when content is wider/taller than the canvas.
    render_w = auto_w
    render_h = auto_h

    img = Image.new("L", (render_w, render_h), bg_color)
    draw = ImageDraw.Draw(img)

    x0 = pad
    y0 = pad

    for row_idx, line in enumerate(lines):
        y = y0 + row_idx * char_h
        draw.text((x0, y), line, fill=text_color, font=font)

    target_w = width if width is not None else render_w
    target_h = height if height is not None else render_h
    if (target_w, target_h) != (render_w, render_h):
        scale = min(target_w / render_w, target_h / render_h)
        fit_w = max(1, int(render_w * scale))
        fit_h = max(1, int(render_h * scale))
        img = img.resize((fit_w, fit_h), Image.LANCZOS)
        canvas = Image.new("L", (target_w, target_h), bg_color)
        paste_x = (target_w - fit_w) // 2
        paste_y = (target_h - fit_h) // 2
        canvas.paste(img, (paste_x, paste_y))
        img = canvas

    arr = np.array(img, dtype=np.float64) / 255.0
    if invert:
        arr = 1.0 - arr
    logger.info("Rendered ASCII art image: (%d x %d)", target_w, target_h)
    return arr


# ---------------------------------------------------------------------------
# 4. QR code renderer
# ---------------------------------------------------------------------------


def render_qr(
    data: str,
    width: int | None = None,
    height: int | None = None,
    logo_path: str | None = None,
    invert: bool = False,
) -> np.ndarray:
    """Render a QR code as a grayscale numpy array.

    Uses the qrcode library with high error correction (ERROR_CORRECT_H) so
    that a logo overlay is recoverable.

    Args:
        data: The string to encode in the QR code.
        width: Optional output width in pixels. If None, uses native QR size.
        height: Optional output height in pixels. If None, uses native QR size.
        logo_path: Optional path to an image to overlay at the center of the
            QR code (resized to 25% of QR dimensions).
        invert: If True, invert pixel values.

    Returns:
        2D float64 array with values in [0.0, 1.0].

    Raises:
        ImportError: If the qrcode library is not installed.
    """
    try:
        import qrcode
        import qrcode.constants
    except ImportError as exc:
        raise ImportError(
            "QR rendering requires qrcode. Install: pip install qrcode[pil]"
        ) from exc

    logger.info("Rendering QR code for data of length %d", len(data))
    qr = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGBA")

    if logo_path is not None:
        logo_p = Path(logo_path)
        if not logo_p.exists():
            raise FileNotFoundError(f"Logo file not found: {logo_path}")
        with Image.open(logo_p) as logo_img:
            logo_rgba = logo_img.convert("RGBA")
        qr_w, qr_h = qr_img.size
        logo_w = qr_w // 4
        logo_h = qr_h // 4
        logo_rgba = logo_rgba.resize((logo_w, logo_h), Image.LANCZOS)
        paste_x = (qr_w - logo_w) // 2
        paste_y = (qr_h - logo_h) // 2
        qr_img.paste(logo_rgba, (paste_x, paste_y), logo_rgba)
        logger.debug("Pasted logo onto QR code at (%d, %d)", paste_x, paste_y)

    qr_gray = qr_img.convert("L")

    # Fit the QR code into the target canvas while preserving its square aspect
    # ratio, then center it on a white background.  Keeping the QR square is
    # important: non-uniform scaling produces rectangular modules whose
    # appearance in a spectrogram depends on the display aspect ratio in
    # unpredictable ways.  The --verify-after-encode spectrogram is generated
    # with a height chosen so that each encoded image pixel appears square
    # (see cmd_encode), which means a square QR canvas patch will appear as a
    # square in the verification PNG.  LANCZOS gives smooth anti-aliased module
    # edges without the harsh aliasing of NEAREST at non-integer scale factors.
    target_w = width if width is not None else qr_gray.width
    target_h = height if height is not None else qr_gray.height
    if (target_w, target_h) != qr_gray.size:
        scale = min(target_w / qr_gray.width, target_h / qr_gray.height)
        fit_w = max(1, int(qr_gray.width * scale))
        fit_h = max(1, int(qr_gray.height * scale))
        qr_scaled = qr_gray.resize((fit_w, fit_h), Image.LANCZOS)
        canvas = Image.new("L", (target_w, target_h), 255)
        paste_x = (target_w - fit_w) // 2
        paste_y = (target_h - fit_h) // 2
        canvas.paste(qr_scaled, (paste_x, paste_y))
        qr_gray = canvas

    arr = np.array(qr_gray, dtype=np.float64) / 255.0
    if invert:
        arr = 1.0 - arr
    return arr


# ---------------------------------------------------------------------------
# 5. SVG renderer
# ---------------------------------------------------------------------------


def render_svg(
    path: str,
    width: int | None = None,
    height: int | None = None,
    dpi: int = 96,
    invert: bool = False,
) -> np.ndarray:
    """Render an SVG file as a grayscale numpy array using cairosvg.

    Args:
        path: Path to the SVG file.
        width: Optional output width in pixels.
        height: Optional output height in pixels.
        dpi: Dots per inch for rendering (default 96).
        invert: If True, invert pixel values.

    Returns:
        2D float64 array with values in [0.0, 1.0].

    Raises:
        RuntimeError: If cairosvg is not installed.
        FileNotFoundError: If the SVG file does not exist.
    """
    try:
        import cairosvg
    except ImportError as exc:
        raise RuntimeError(
            "SVG rendering requires cairosvg. Install: pip install cairosvg"
        ) from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"SVG file not found: {path}")

    logger.info("Rendering SVG: %s (dpi=%d)", path, dpi)

    kwargs: dict = {"url": str(p), "dpi": dpi}
    if width is not None:
        kwargs["output_width"] = width
    if height is not None:
        kwargs["output_height"] = height

    png_bytes = cairosvg.svg2png(**kwargs)
    img = Image.open(BytesIO(png_bytes)).convert("L")

    arr = np.array(img, dtype=np.float64) / 255.0
    if invert:
        arr = 1.0 - arr
    return arr


# ---------------------------------------------------------------------------
# 7. Math function renderer
# ---------------------------------------------------------------------------

_ALLOWED_MATH_FUNCTIONS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sqrt": math.sqrt,
    "abs": abs,
    "exp": math.exp,
    "log": math.log,
}


class _SafeMathEvaluator(ast.NodeVisitor):
    """AST visitor that safely evaluates a single-variable math expression.

    Only allows: BinOp (+,-,*,/,**), UnaryOp (unary minus/plus), Call
    (whitelisted functions), Constant, and Name (only 'x').
    """

    def __init__(self, x_value: float) -> None:
        self.x = x_value

    def visit(self, node: ast.AST) -> float:  # type: ignore[override]
        return super().visit(node)

    def visit_Expression(self, node: ast.Expression) -> float:
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> float:
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = node.op
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            return left / right
        if isinstance(op, ast.Pow):
            return left ** right
        if isinstance(op, ast.Mod):
            return left % right
        raise ValueError(f"Unsupported binary operator: {type(op).__name__}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        operand = self.visit(node.operand)
        op = node.op
        if isinstance(op, ast.USub):
            return -operand
        if isinstance(op, ast.UAdd):
            return +operand
        raise ValueError(f"Unsupported unary operator: {type(op).__name__}")

    def visit_Call(self, node: ast.Call) -> float:
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed.")
        func_name = node.func.id
        if func_name not in _ALLOWED_MATH_FUNCTIONS:
            raise ValueError(
                f"Function '{func_name}' is not allowed. "
                f"Allowed: {list(_ALLOWED_MATH_FUNCTIONS)}"
            )
        fn = _ALLOWED_MATH_FUNCTIONS[func_name]
        args = [self.visit(arg) for arg in node.args]
        return fn(*args)

    def visit_Constant(self, node: ast.Constant) -> float:
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    def visit_Name(self, node: ast.Name) -> float:
        if node.id == "x":
            return self.x
        if node.id == "pi":
            return math.pi
        if node.id == "e":
            return math.e
        raise ValueError(
            f"Unknown name '{node.id}'. Only 'x', 'pi', and 'e' are allowed."
        )

    def generic_visit(self, node: ast.AST) -> float:  # type: ignore[override]
        raise ValueError(f"Disallowed AST node type: {type(node).__name__}")


def _eval_math_expr(expression: str, x: float) -> float:
    """Safely evaluate a math expression for a given x value.

    Args:
        expression: A mathematical expression string using variable 'x'.
        x: The value of x.

    Returns:
        The computed float result.

    Raises:
        ValueError: If the expression contains disallowed constructs.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {exc}") from exc
    evaluator = _SafeMathEvaluator(x)
    return evaluator.visit(tree)


def render_math(
    expression: str,
    width: int = 512,
    height: int = 256,
    x_range: tuple[float, float] = (-10.0, 10.0),
    thickness: int = 3,
    invert: bool = False,
) -> np.ndarray:
    """Render a math function y=f(x) as a grayscale plot.

    The expression is parsed safely using Python's ast module — no eval() on
    raw strings. Supported functions: sin, cos, tan, sqrt, abs, exp, log.
    Supported constants: pi, e. Supported operators: +, -, *, /, **, %.

    Args:
        expression: A math expression string, e.g. "sin(x) * exp(-x**2 / 10)".
        width: Output width in pixels (default 512).
        height: Output height in pixels (default 256).
        x_range: (x_min, x_max) range for the x-axis (default (-10.0, 10.0)).
        thickness: Line thickness in pixels (default 3).
        invert: If True, invert pixel values (dark background, bright curve).

    Returns:
        2D float64 array with values in [0.0, 1.0].

    Raises:
        ValueError: If the expression contains disallowed constructs.
    """
    logger.info("Rendering math expression: '%s' over x in %s", expression, x_range)

    x_min, x_max = x_range
    xs = np.linspace(x_min, x_max, width)

    ys: list[float | None] = []
    for x_val in xs:
        try:
            y_val = _eval_math_expr(expression, float(x_val))
            if not math.isfinite(y_val):
                ys.append(None)
            else:
                ys.append(y_val)
        except (ValueError, ZeroDivisionError, OverflowError):
            ys.append(None)

    valid_ys = [y for y in ys if y is not None]
    if not valid_ys:
        raise ValueError("Expression produced no finite values over the given x_range.")

    y_min = min(valid_ys)
    y_max = max(valid_ys)
    y_span = y_max - y_min
    if y_span == 0.0:
        y_span = 1.0
    pad = y_span * 0.10
    y_min -= pad
    y_max += pad
    y_span = y_max - y_min

    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)

    def to_pixel(x_idx: int, y_val: float) -> tuple[int, int]:
        px = x_idx
        py = int((1.0 - (y_val - y_min) / y_span) * (height - 1))
        py = max(0, min(height - 1, py))
        return px, py

    points: list[tuple[int, int]] = []
    for i, y_val in enumerate(ys):
        if y_val is not None:
            points.append(to_pixel(i, y_val))
        else:
            if len(points) >= 2:
                draw.line(points, fill=0, width=thickness)
            points = []

    if len(points) >= 2:
        draw.line(points, fill=0, width=thickness)
    elif len(points) == 1:
        px, py = points[0]
        r = thickness // 2
        draw.ellipse([px - r, py - r, px + r, py + r], fill=0)

    arr = np.array(img, dtype=np.float64) / 255.0
    if invert:
        arr = 1.0 - arr
    return arr


# ---------------------------------------------------------------------------
# 8. Image sequence / animated GIF renderer
# ---------------------------------------------------------------------------


def render_sequence(
    paths: list[str] | None = None,
    gif_path: str | None = None,
    frame_width: int = 128,
    height: int = 128,
    invert: bool = False,
) -> np.ndarray:
    """Render an image sequence or animated GIF as a wide concatenated array.

    Frames are concatenated horizontally, producing an array of shape
    (height, frame_width * num_frames).

    Exactly one of paths or gif_path must be provided.

    Args:
        paths: List of file paths, one per frame.
        gif_path: Path to an animated GIF file.
        frame_width: Width of each individual frame in pixels (default 128).
        height: Height of each frame in pixels (default 128).
        invert: If True, invert pixel values.

    Returns:
        2D float64 array of shape (height, frame_width * num_frames) with
        values in [0.0, 1.0].

    Raises:
        ValueError: If neither or both of paths and gif_path are provided.
        FileNotFoundError: If any specified file does not exist.
    """
    if paths is None and gif_path is None:
        raise ValueError("Provide either paths or gif_path.")
    if paths is not None and gif_path is not None:
        raise ValueError("Provide either paths or gif_path, not both.")

    frames: list[np.ndarray] = []

    if gif_path is not None:
        p = Path(gif_path)
        if not p.exists():
            raise FileNotFoundError(f"GIF file not found: {gif_path}")
        with Image.open(p) as gif:
            frame_idx = 0
            while True:
                try:
                    gif.seek(frame_idx)
                except EOFError:
                    break
                frame_img = gif.convert("L").resize((frame_width, height), Image.LANCZOS)
                arr = np.array(frame_img, dtype=np.float64) / 255.0
                frames.append(arr)
                frame_idx += 1
        logger.info("Extracted %d frames from GIF: %s", len(frames), gif_path)
    else:
        for fp in paths:  # type: ignore[union-attr]
            fpath = Path(fp)
            if not fpath.exists():
                raise FileNotFoundError(f"Frame file not found: {fp}")
            with Image.open(fpath) as img:
                frame_img = img.convert("L").resize((frame_width, height), Image.LANCZOS)
            arr = np.array(frame_img, dtype=np.float64) / 255.0
            frames.append(arr)
        logger.info("Loaded %d frames from file list.", len(frames))

    if not frames:
        raise ValueError("No frames found in the provided input.")

    result = np.concatenate(frames, axis=1)
    if invert:
        result = 1.0 - result
    return result
