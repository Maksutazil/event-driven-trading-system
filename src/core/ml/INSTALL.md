# Installing the Machine Learning Module

This guide will help you set up the machine learning components for the event-driven trading system.

## Dependencies

The ML module requires the following Python packages:

```
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
joblib>=1.1.0
matplotlib>=3.4.0  # For visualizations
```

These dependencies are used for:
- **NumPy** and **Pandas**: Data manipulation and numerical operations
- **scikit-learn**: Machine learning models and metrics
- **joblib**: Model serialization/deserialization
- **matplotlib**: Visualization of model performance (optional)

## Installation

### 1. Install dependencies

You can install all required dependencies using pip:

```bash
pip install numpy pandas scikit-learn joblib matplotlib
```

Or use the provided requirements file:

```bash
pip install -r requirements.txt
```

### 2. Verify installation

You can verify the installation by running the test script:

```bash
python test_model_manager.py
```

This script will:
1. Create a synthetic dataset
2. Train and save a simple model
3. Load the model with the ModelManager
4. Register a transformer
5. Make predictions and update the model
6. Run various tests to verify functionality

If all tests pass, the ML module is correctly installed.

## Optional: GPU Support

For better performance with large models, you can install the GPU-accelerated version of scikit-learn via:

```bash
pip install scikit-learn-intelex
```

## Adding New Model Types

The ML module currently supports scikit-learn models by default. To add support for other model types:

1. Create a new adapter in `src/core/ml/adapters/`
2. Implement the `Model` interface for your model type
3. Register your adapter in the `_get_model_adapter` method in `DefaultModelManager`

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'sklearn'**
   - Solution: Ensure scikit-learn is installed: `pip install scikit-learn`

2. **Module not found errors**
   - Solution: Make sure the project root is in your Python path

3. **Thread-related errors**
   - Solution: Update to the latest version of your Python interpreter

### Getting Help

If you encounter issues with the ML module, please:

1. Check the logs for detailed error messages
2. Review the error handling guide in README.md
3. Contact the development team with the error details 