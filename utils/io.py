import os, re, yaml, base64

DELIMS = re.compile(r"\n-{3,}\n|\r?\n\r?\n")

def load_questions_from_text(txt: str):
    try:
        y = yaml.safe_load(txt)
        if isinstance(y, list) and all(isinstance(q, str) for q in y):
            return [q.strip() for q in y if q and q.strip()]
    except Exception:
        pass
    parts = [p.strip() for p in re.split(DELIMS, txt) if p.strip()]
    if not parts and txt.strip():
        parts = [txt.strip()]
    if len(parts) == 1:
        lines = [re.sub(r'^[\-\*\d\.\s]+', '', ln.strip()) for ln in parts[0].splitlines() if ln.strip()]
        return lines
    return parts

def detect_output_format(txt: str):
    s = txt.lower()
    if "respond with a json array" in s or "return with a json array" in s or "return a json array" in s:
        return "array"
    if "respond with a json object" in s or "return with a json object" in s or "return a json object" in s:
        return "object"
    # heuristic: presence of object example
    if "{" in txt and "}" in txt and ":" in txt:
        return "object"
    return "array"

def data_uri_png(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8")
