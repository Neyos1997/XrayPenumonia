from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
import cv2
import os
import logging
from langchain.llms import CTransformers
from langchain import PromptTemplate, LLMChain

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__, template_folder=r'yourlocation')
app.config['UPLOAD_FOLDER'] = 'uploads'

# Set the location to load the CNN model
cnn_model_location = 'yourlocation'
if os.path.exists(cnn_model_location):
    cnn_model = load_model(cnn_model_location)
    logging.info("Loaded CNN model from disk")
else:
    logging.error("CNN Model not found. Please train the model first.")

# Initialize LLM model
llm = CTransformers(model="TheBloke/Llama-2-13B-Ensemble-v5-GGUF",
                    model_file="llama-2-13b-ensemble-v5.Q2_K.gguf",
                    model_type="llama")

# Define template
template = """
              Describe the findings in the following image: `{image_name}`.
              Condition: Pneumonia
              Probability of Pneumonia (CNN Model): {probability}
              Opacity Degree of Lungs: {opacity_degree:.2f}%
              Return your response in bullet points which covers the key points of the findings.
              ```{text}```
              interpret these results to confirm or deny if it is a case of pneumonia.
              BULLET POINT SUMMARY:
              - Condition: Pneumonia
              - Probability of Pneumonia (CNN Model): {probability}
              - Opacity Degree of Lungs: {opacity_degree:.2f}%
           """

# Initialize LLMChain
prompt = PromptTemplate(template=template, input_variables=["image_name", "probability", "opacity_degree", "text"])
llm_chain = LLMChain(prompt=prompt, llm=llm)


# Function to contour lungs and calculate opacity degree
def contour_lungs(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    _, threshold_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros_like(img)
    cv2.drawContours(contour_mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    contour_mask_gray = cv2.cvtColor(contour_mask, cv2.COLOR_BGR2GRAY)
    total_area = img.shape[0] * img.shape[1]
    highlighted_area = cv2.countNonZero(contour_mask_gray)
    opacity_degree = (highlighted_area / total_area) * 100
    contoured_img = cv2.bitwise_and(img, img, mask=contour_mask_gray)
    contoured_img[:, :, 0] = np.where(contour_mask_gray == 255, 255, contoured_img[:, :, 0])  # Highlight opacity in blue color
    contoured_img[:, :, 1] = np.where(contour_mask_gray == 255, 0, contoured_img[:, :, 1])
    contoured_img[:, :, 2] = np.where(contour_mask_gray == 255, 0, contoured_img[:, :, 2])
    return contoured_img, opacity_degree


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        # Read the uploaded image
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        contoured_img, opacity_degree = contour_lungs(image)

        # Resize the processed image to match the input shape of the model (if necessary)
        resized_img = cv2.resize(contoured_img, (150, 150))

        # Predict using the pre-trained CNN model
        probability = cnn_model.predict(np.expand_dims(resized_img, axis=0))[0][0]

        # Convert probability to Python float
        probability = float(probability)

        # Generate NLP interpretation
        nlp_interpretation = llm_chain.run(
            {'image_name': file.filename, 'probability': probability, 'opacity_degree': opacity_degree, 'text': ''})

        # Debugging: Print processed image path
        processed_img_name = 'processed_' + secure_filename(file.filename)
        processed_img_path = os.path.join('yourloction', processed_img_name)
        print("Processed image path:", processed_img_path)

        # Save the processed image
        cv2.imwrite(processed_img_path, contoured_img)

        return jsonify({'processedImagePath': processed_img_path,
                        'probability': probability,
                        'opacityPercentage': opacity_degree,
                        'nlpInterpretation': nlp_interpretation})

    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=False)
