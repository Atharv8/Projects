from flask import Flask, render_template, request, send_file
import cv2
import easyocr
import os
import numpy
from flask import send_from_directory

app = Flask(__name__)

download_directory = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "download")
)


def process_video(video_path):
    reader = easyocr.Reader(["en"], gpu=True)
    video_cap = cv2.VideoCapture(video_path)

    text_op = ""

    while video_cap.isOpened():
        ret, frame = video_cap.read()

        if not ret:
            break

        resized_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        result = reader.readtext(resized_frame)

        for detection in result:
            text_op += detection[1] + "\n"

    video_cap.release()

    output_txt_file = os.path.join(download_directory, "extracted_text.txt")

    with open(output_txt_file, "w") as text_file:
        text_file.write(text_op)

    return output_txt_file


def processimage(image):
    reader = easyocr.Reader(["en"], gpu=True)
    text = reader.readtext(image)
    lis = [t[1] for t in text if t[2] > 0.25]
    return "\n".join(lis)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/image_extractor")
def image_extractor():
    return render_template("image_extractor.html")  # Render the image extractor page


@app.route("/video_extractor")
def video_extractor():
    return render_template("video_extractor.html")  # Render the video extractor page


@app.route("/upload_video", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        return render_template("index.html", message="No file part")

    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html", message="No selected file")

    if file:
        file_path = os.path.join("static/uploads", file.filename)
        file.save(file_path)

        # Process the uploaded video
        processed_text_path = process_video(file_path)

        return send_file(
            processed_text_path,
            as_attachment=True,
        )

    return render_template("index.html")


@app.route("/upload_image", methods=["POST"])
def upload():
    if "file" not in request.files:
        return render_template("index.html", message="No file part")

    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html", message="No selected file")

    if file:
        npimg = numpy.fromstring(file.read(), numpy.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        lis = processimage(img)

        temp_file = "extracted_text.txt"
        with open(temp_file, "w") as f:
            f.write(lis)

        return send_file(temp_file, as_attachment=True)
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
