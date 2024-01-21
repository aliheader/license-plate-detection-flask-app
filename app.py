from flask import Flask, render_template, request
import base64
import io

from detect.detect import detect

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    original_image_data = None
    processed_image_data = None
    if request.method == "POST":
        # Get the uploaded image from the form
        uploaded_file = request.files["image"]

        if uploaded_file:
            file = uploaded_file.read()
            processed_img = detect(file)

            print(processed_img)

            org_img_base64 = base64.b64encode(file).decode("utf-8")

            with io.BytesIO() as byte_stream:
                processed_img.save(byte_stream, format="JPEG")
                byte_data = byte_stream.getvalue()

            processed_img_base64 = base64.b64encode(byte_data).decode("utf-8")

            # Display the processed image in the HTML template
            return render_template(
                "index.html",
                original_image_data=org_img_base64,
                processed_image_data=processed_img_base64,
            )

    return render_template(
        "index.html",
        original_image_data=original_image_data,
        processed_image_data=processed_image_data,
    )


if __name__ == "__main__":
    app.run(debug=True)
