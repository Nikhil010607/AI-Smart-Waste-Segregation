
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Get the project folder path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained CNN model
MODEL_PATH = os.path.join(BASE_DIR, "waste_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# Waste categories (must match the training class order)
classes = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# Upload folder
UPLOAD_FOLDER = os.path.join(
    BASE_DIR, "static", "uploads"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Allowed image formats
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")

        if not file or file.filename == "":
            error = "Please select an image."

        elif not allowed_file(file.filename):
            error = "Please upload a JPG, JPEG, or PNG image."

        else:
            filename = secure_filename(file.filename)

            # Prevent overwriting existing uploads
            name, extension = os.path.splitext(filename)
            filename = f"{name}_{os.urandom(4).hex()}{extension}"

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            file.save(filepath)

            # Read uploaded image
            img = cv2.imread(filepath)

            if img is None:
                error = "Unable to read the image. Please try another."

            else:
                try:
                    # Preprocess image
                    # Assumes training used RGB images, 224x224,
                    # and pixel values normalized to 0-1.
                    img = cv2.cvtColor(
                        img, cv2.COLOR_BGR2RGB
                    )

                    img = cv2.resize(img, (224, 224))
                    img = img.astype("float32") / 255.0
                    img = np.expand_dims(img, axis=0)

                    # Predict waste category
                    pred = model.predict(img, verbose=0)

                    predicted_index = int(np.argmax(pred))
                    prediction = classes[predicted_index]

                    # Path used by the HTML template
                    image_path = "uploads/" + filename

                except Exception as e:
                    app.logger.exception("Prediction failed")
                    error = "Prediction failed. Please check the model and preprocessing."

    return render_template(
        "index.html",
        prediction=prediction,
        image_path=image_path,
        error=error
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
