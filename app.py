import os
import io
import base64
import pickle

import matplotlib
matplotlib.use("Agg")  # headless-safe backend for servers like Render
import matplotlib.pyplot as plt

from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "vqc_model.pkl")

with open(MODEL_PATH, "rb") as f:
    vqc_model = pickle.load(f)


def make_chart(feature1: float, feature2: float, prediction: float) -> str:
    """Render a small, on-brand result chart and return it as a base64 PNG string."""
    is_up = prediction >= 0
    bar_color = "#0f9d58" if is_up else "#e5484d"

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(5, 3.6), dpi=150)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    ax.bar(["Prediction"], [prediction], color=bar_color, width=0.45, zorder=3)
    ax.axhline(0, color="#94a3b8", linewidth=1)

    ax.set_title("Predicted Value", fontsize=12, color="#0f172a", pad=12, fontweight="bold")
    ax.set_ylabel("Model output", fontsize=9, color="#5b6b82")
    ax.tick_params(colors="#5b6b82", labelsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#e2e8f0")

    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8, zorder=0)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    graph_url = None
    error = None
    feature1 = None
    feature2 = None

    if request.method == "POST":
        raw1 = request.form.get("feature1", "").strip()
        raw2 = request.form.get("feature2", "").strip()

        try:
            feature1 = float(raw1)
            feature2 = float(raw2)
        except (TypeError, ValueError):
            error = "Please enter valid numbers for both indicators."
        else:
            try:
                prediction = float(vqc_model.predict([[feature1, feature2]])[0])
                graph_url = make_chart(feature1, feature2, prediction)
            except Exception:
                error = "The model couldn't process these inputs. Please try different values."
                prediction = None

    return render_template(
        "predict.html",
        prediction=prediction,
        graph_url=graph_url,
        error=error,
        feature1=feature1,
        feature2=feature2,
    )


# ---- Render / external deploy ----
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
