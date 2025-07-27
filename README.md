# Iris Flower Classification Web Interface

## Overview
This is a beautiful, interactive web application for classifying Iris flowers using a neural network. The interface features modern design, animations, and comprehensive visualizations.

## Features
- 🌸 **Interactive Prediction**: Input flower measurements using sliders
- 📊 **Beautiful Visualizations**: Confidence scores, radar charts, and 3D scatter plots
- 🎨 **Modern Design**: Gradient backgrounds, animations, and responsive layout
- 📈 **Real-time Results**: Instant predictions with confidence percentages
- 🌺 **Species Information**: Detailed descriptions of each Iris species

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Required Packages
```bash
pip install streamlit torch pandas numpy plotly scikit-learn matplotlib
```

## Running the Application

1. **Start the Streamlit server:**
   ```bash
   streamlit run iris_web_app.py
   ```

2. **Open your browser and navigate to:**
   ```
   http://localhost:8501
   ```

## How to Use

1. **Adjust the sliders** in the sidebar to input flower measurements:
   - 🌿 Sepal Length (4.0 - 8.0 cm)
   - 🌿 Sepal Width (2.0 - 4.5 cm)
   - 🌺 Petal Length (1.0 - 7.0 cm)
   - 🌺 Petal Width (0.1 - 2.5 cm)

2. **Click "🔮 Predict Species"** to get the classification result

3. **View the results:**
   - Predicted species with confidence percentage
   - Confidence scores bar chart
   - Radar chart showing your flower's characteristics
   - Species information and description

4. **Explore the dataset:**
   - 3D visualization of the Iris dataset
   - Dataset statistics
   - Model information in the expandable section

## Application Structure

```
iris_web_app.py          # Main Streamlit application
iris_classifier_code.py  # Neural network training code
README.md               # This file
```

## Technical Details

### Neural Network Architecture
- **Input Layer**: 4 features (sepal length, sepal width, petal length, petal width)
- **Hidden Layer 1**: 8 neurons with ReLU activation
- **Hidden Layer 2**: 9 neurons with ReLU activation
- **Output Layer**: 3 neurons (one for each species)

### Iris Species
- **Setosa** 🌸: Characterized by smaller petals
- **Versicolor** 🌼: Medium-sized petals and sepals
- **Virginica** 🌺: Typically has the largest petals and sepals

## Customization

You can customize the application by modifying:
- **Colors**: Update the CSS gradients in the `st.markdown()` sections
- **Model**: Replace the neural network with your own trained model
- **Visualizations**: Modify the Plotly charts for different representations
- **Layout**: Adjust the Streamlit layout and components

## Troubleshooting

### Common Issues

1. **Port already in use:**
   ```bash
   streamlit run iris_web_app.py --server.port 8502
   ```

2. **Missing packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Model loading errors:**
   - Ensure the neural network class is properly defined
   - Check that all dependencies are installed

## Performance

The application is optimized for:
- Fast prediction responses
- Smooth animations
- Responsive design for different screen sizes
- Efficient data caching with Streamlit decorators

## Future Enhancements

Potential improvements:
- Add model training interface
- Include more flower species
- Implement batch prediction
- Add data upload functionality
- Include model performance metrics

---

**Created with ❤️ using Streamlit and PyTorch**
