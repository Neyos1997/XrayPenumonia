import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from keras.models import load_model
from langchain.llms import CTransformers
from langchain import PromptTemplate, LLMChain


input_path = r'yourlocation'


model_location = r'yourlocation'


if os.path.exists(model_location):
    model = load_model(model_location)
    print("Loaded CNN model from disk")
else:
    print("CNN Model not found. Please train the model first.")

# Function to process test data with NLP analysis
def process_test_data_with_nlp(img_dim):
    test_data = []
    image_paths = []
    for class_folder in ['NORMAL', 'PNEUMONIA']:
        class_path = os.path.join(input_path, class_folder)
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            img = cv2.resize(cv2.imread(img_path, cv2.IMREAD_COLOR), (img_dim, img_dim))
            test_data.append(img)
            image_paths.append(img_path)
    return test_data, image_paths

def contour_lungs(img):
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # Apply adaptive thresholding to segment the lungs
    _, threshold_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask to draw the contours
    contour_mask = np.zeros_like(img)

    # Draw contours on the mask
    cv2.drawContours(contour_mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # Convert the mask to grayscale
    contour_mask_gray = cv2.cvtColor(contour_mask, cv2.COLOR_BGR2GRAY)

    # Calculate the opacity degree based on the contour area
    total_area = img.shape[0] * img.shape[1]
    highlighted_area = cv2.countNonZero(contour_mask_gray)
    opacity_degree = (highlighted_area / total_area) * 100  # Opacity as a percentage

    # Apply the mask to the original image to get the contoured image
    contoured_img = cv2.bitwise_and(img, img, mask=contour_mask_gray)

    return contoured_img, opacity_degree


 

def generate_nlp_interpretation(img_name, probability, opacity_degree):
    # Generate textual summary for the finding
    text = f"Probability of Pneumonia: {probability}\nOpacity Degree of Lungs: {opacity_degree:.2f}%"
    return text

# Initialize LLM model
llm = CTransformers(model="TheBloke/Llama-2-13B-Ensemble-v5-GGUF", model_file="llama-2-13b-ensemble-v5.Q2_K.gguf", model_type="llama")

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


prompt = PromptTemplate(template=template, input_variables=["image_name", "probability", "opacity_degree", "text"])

# Initialize LLMChain
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Set image dimensions
img_dims = 150

# Process the test data and generate NLP interpretations
test_data, image_paths = process_test_data_with_nlp(img_dims)

# Check if test_data is empty
if not test_data:
    print("No test data found. Please check your data loading logic.")
else:
    # Analyze images and generate interpretations
    for img, img_path in zip(test_data, image_paths):
        img_name = os.path.basename(img_path)
        print(f"Image Name: {img_name}")
        
        # Predict using pre-trained CNN model
        probability = model.predict(np.array([img]))[0][0]
        
        # Contour lungs and calculate opacity degree
        contoured_img, opacity_degree = contour_lungs(img)
        
        # Display the original and contoured images side by side
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Highlight opacity in red on the contoured image
        contoured_with_opacity = np.copy(contoured_img)
        contoured_with_opacity[:, :, 0] = np.where(contoured_with_opacity[:, :, 0] == 0, 0, 255)
        axes[1].imshow(cv2.cvtColor(contoured_with_opacity, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Contoured Lungs (Opacity Highlighted)")
        axes[1].axis('off')
        
        plt.show()
        
        # Print opacity degree, probability, and image name
        print(f"Opacity Degree of Lungs: {opacity_degree:.2f}%")
        print(f"Probability of Pneumonia: {probability}")
        
        # Generate NLP interpretation
        nlp_interpretation = llm_chain.run({'image_name': img_name, 'probability': probability, 'opacity_degree': opacity_degree, 'text': ''})

        print("Findings Summary:")
        print(nlp_interpretation)
