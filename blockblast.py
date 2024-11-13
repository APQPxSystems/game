# Extracting game state from the screenshot
import cv2
import numpy as np

def extract_grid_from_screenshot(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert to grayscale (simplifies image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to make block colors distinct
    _, thresholded = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    # Find contours of the blocks (this is where game pieces will be located)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours, img

# Identify block colors
def get_block_colors(contours, img):
    colors = []
    for contour in contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Extract the color from the image
        block_img = img[y:y+h, x:x+w]
        avg_color = np.mean(block_img, axis=(0, 1))  # Average color of the block

        # Convert average color to a more readable format (RGB)
        colors.append(tuple(int(c) for c in avg_color))
    
    return colors

# Game state analysis and move recommendations
def analyze_game_state(colors):
    # Simple strategy (example): Find a match of colors and suggest moves
    # In a more sophisticated version, this would involve analyzing the best possible moves

    color_counts = {color: colors.count(color) for color in set(colors)}
    
    # If there are more than two blocks of the same color adjacent, suggest a move
    suggestions = []
    for color, count in color_counts.items():
        if count >= 3:  # Example condition for recommending a move
            suggestions.append(f"Consider clearing {color} blocks.")
    
    if not suggestions:
        suggestions.append("No immediate matches found, try to clear blocks strategically.")
    
    return suggestions

# Streamlit interface
import streamlit as st
from PIL import Image

def main():
    st.title('BlockBlast Game Hack')
    
    # Upload the screenshot
    uploaded_file = st.file_uploader("Upload a screenshot of your BlockBlast game", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Display the image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Screenshot", use_column_width=True)

        # Process the screenshot (convert to OpenCV format)
        img_cv = np.array(img)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        # Extract game state (contours and colors)
        contours, img_with_contours = extract_grid_from_screenshot(uploaded_file)
        colors = get_block_colors(contours, img_with_contours)
        
        # Analyze the game state and provide recommendations
        recommendations = analyze_game_state(colors)
        
        # Display recommendations
        st.write("### Recommendations:")
        for recommendation in recommendations:
            st.write(f"- {recommendation}")

if __name__ == "__main__":
    main()
