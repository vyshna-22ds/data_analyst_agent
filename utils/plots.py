import io, numpy as np, pandas as pd
import matplotlib.pyplot as plt

def to_png(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0); return buf.getvalue()

def _downsize(render_fn, max_bytes=100_000, dpi=120, steps=4):
    png = render_fn(dpi)
    tries = 0
    while len(png) > max_bytes and dpi > 60 and tries < steps:
        dpi = max(60, dpi - 20)
        png = render_fn(dpi)
        tries += 1
    return png

def scatter(df, x, y, regression=False, title=None, max_bytes=100_000):
    xv = pd.to_numeric(df[x], errors="coerce").astype(float)
    yv = pd.to_numeric(df[y], errors="coerce").astype(float)
    m = xv.notna() & yv.notna()
    xv, yv = xv[m].values, yv[m].values
    if xv.size == 0 or yv.size == 0:
        raise ValueError("No finite data for scatter.")
    # fit y = a + b x
    if regression:
        b, a = np.polyfit(xv, yv, 1)
    def render(dpi):
        fig = plt.figure(figsize=(4,3), dpi=dpi); ax = fig.gca()
        ax.scatter(xv, yv, s=12)
        if regression:
            xs = np.linspace(float(np.min(xv)), float(np.max(xv)), 100)
            ys = a + b*xs
            ax.plot(xs, ys, linestyle=":", color="red", linewidth=1.4)  # dotted red
        ax.set_xlabel(x); ax.set_ylabel(y)
        if title: ax.set_title(title)
        return to_png(fig, dpi)
    png = _downsize(render, max_bytes=max_bytes)
    return png

def line(df, x, y, title=None, max_bytes=100_000):
    xv = df[x]; yv = df[y]
    def render(dpi):
        fig = plt.figure(figsize=(4,3), dpi=dpi); ax = fig.gca()
        ax.plot(xv, yv)
        ax.set_xlabel(x); ax.set_ylabel(y)
        if title: ax.set_title(title)
        return to_png(fig, dpi)
    return _downsize(render, max_bytes=max_bytes)

def bar(df, x, y, title=None, max_bytes=100_000):
    xv = df[x].astype(str); yv = pd.to_numeric(df[y], errors="coerce")
    def render(dpi):
        fig = plt.figure(figsize=(4,3), dpi=dpi); ax = fig.gca()
        ax.bar(xv, yv)
        ax.set_xlabel(x); ax.set_ylabel(y); ax.tick_params(axis='x', labelrotation=45)
        if title: ax.set_title(title)
        return to_png(fig, dpi)
    return _downsize(render, max_bytes=max_bytes)

def pie(df, label, value, title=None, max_bytes=100_000):
    lv = df[label].astype(str); vv = pd.to_numeric(df[value], errors="coerce").fillna(0)
    def render(dpi):
        fig = plt.figure(figsize=(4,3), dpi=dpi); ax = fig.gca()
        ax.pie(vv, labels=lv, autopct='%1.1f%%')
        if title: ax.set_title(title)
        return to_png(fig, dpi)
    return _downsize(render, max_bytes=max_bytes)

def hist(df, col, bins=20, title=None, max_bytes=100_000):
    vv = pd.to_numeric(df[col], errors="coerce").dropna()
    def render(dpi):
        fig = plt.figure(figsize=(4,3), dpi=dpi); ax = fig.gca()
        ax.hist(vv, bins=bins)
        ax.set_xlabel(col); ax.set_ylabel("count")
        if title: ax.set_title(title)
        return to_png(fig, dpi)
    return _downsize(render, max_bytes=max_bytes)
