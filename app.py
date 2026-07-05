from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import logging
import os
import sys
from functools import wraps
from werkzeug.utils import secure_filename

# ------------------- Flask Setup -------------------
app = Flask(__name__)
app.secret_key = 'crop_disease_detection_secret_key_2024'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['DATABASE'] = 'crop_disease_detection.db'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# ------------------- Logging Setup -------------------
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Pre-load model at startup so errors show early
_predict_image = None
_model = None
_generate_gradcam = None
_ml_startup_error = None

try:
    from utils.predict import predict_image as _predict_image, model as _model
    from utils.gradcam import generate_gradcam as _generate_gradcam
    logger.info(f"Model loaded successfully at startup with Python: {sys.executable}")
except Exception as e:
    logger.error(f"STARTUP ERROR loading model: {e}")
    _ml_startup_error = str(e)

def get_db_connection():
    """Create and return a SQLite database connection."""
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the local database if it does not exist yet."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fullname TEXT NOT NULL,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
        """
    )
    conn.commit()
    cursor.close()
    conn.close()

init_db()


def login_required(view_func):
    """Require login before allowing access to protected views."""
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if "username" not in session:
            flash("Please login first to use Detect.", "warning")
            return redirect(url_for("login", next=request.path))
        return view_func(*args, **kwargs)

    return wrapped_view

# ------------------- Routes -------------------

# -------------------------
# INDEX PAGE
# -------------------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("aboutus.html")

# -------------------------
# SIGNUP PAGE
# -------------------------
@app.route("/signup", methods=["GET","POST"])
def signup():
    if request.method == "POST":
        fullname = request.form["fullname"]
        username = request.form["username"]
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])

        # Save user in database
        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                "INSERT INTO users(fullname,username,email,password) VALUES(?,?,?,?)",
                (fullname, username, email, password)
            )
            conn.commit()
        except sqlite3.IntegrityError:
            flash("Username or email already exists.", "danger")
            return render_template("signup.html")
        finally:
            cursor.close()
            conn.close()

        flash("Account created successfully. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")

# -------------------------
# LOGIN PAGE
# -------------------------
@app.route("/login", methods=["GET","POST"])
def login():
    next_page = request.args.get("next")

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session["username"] = username
            return redirect(next_page or url_for("dashboard"))
        else:
            flash("Invalid username or password.", "danger")

    return render_template("login.html")

# -------------------------
# DASHBOARD PAGE
# -------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")

# -------------------------
# IMAGE PREDICTION
# -------------------------
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    file = request.files.get("image")

    if not file or file.filename == "":
        flash("Please choose an image to upload.", "warning")
        return redirect(url_for("dashboard"))

    if _predict_image is None or _model is None or _generate_gradcam is None:
        logger.error(
            "Prediction skipped because ML stack is unavailable. "
            f"startup_error={_ml_startup_error}; python={sys.executable}"
        )
        flash(
            "Prediction model is not available in this Python environment. "
            f"Current interpreter: {sys.executable}",
            "danger"
        )
        return redirect(url_for("dashboard"))

    try:
        # Sanitize filename to avoid path issues
        filename = secure_filename(file.filename)
        if not filename:
            filename = "uploaded_image.jpg"

        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        result_path = os.path.join(app.config["RESULT_FOLDER"], f"result_{filename}")

        file.save(path)
        logger.info(f"Image saved to {path}")

        disease, confidence = _predict_image(path)
        logger.info(f"Prediction: {disease} ({confidence}%)")

        if confidence > 65:
            # High confidence: Generate GradCAM and show the disease
            _generate_gradcam(_model, path, result_path)
            logger.info(f"GradCAM saved to {result_path}")
            final_disease = disease
            gradcam_url = result_path.replace("\\", "/")
        else:
            # Low confidence: Override message and skip GradCAM
            logger.info("Confidence is 65 or below. Sending rejection message.")
            final_disease = "It was not the crop I had trained on."
            gradcam_url = None

        return render_template(
            "resultpage.html",
            image=path.replace("\\", "/"),
            gradcam=gradcam_url, 
            disease=final_disease,
            confidence=round(confidence, 2)
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        flash(f"Error during prediction: {str(e)}", "danger")
        return redirect(url_for("dashboard"))

# -------------------------
# LOGOUT
# -------------------------
@app.route("/logout")
def logout():
    session.clear()
    return render_template("logout.html")

# ------------------- Run Server -------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)