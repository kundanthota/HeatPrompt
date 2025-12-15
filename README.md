# HeatPrompt

A comprehensive toolkit for predicting building heat demand using satellite imagery, AI-powered caption generation, and machine learning models.

## Overview

This project provides an end-to-end pipeline for analyzing building heat demand using satellite imagery and AI. The workflow includes data collection from the RLP Atlas, image capture, AI caption generation, feature extraction, model training, and a demo application to serve predictions.

## Step-by-Step Implementation Guide

### 1. Data Collection from RLP Atlas

The first step is to fetch geographical data from the RLP Atlas Web Feature Service (WFS) using bounding boxes:

```bash
python 01_fetch_data_bbox.py --bbox minLon,minLat,maxLon,maxLat
```

This script:
- Takes bounding box coordinates as input
- Connects to the RLP Atlas WFS service
- Downloads building footprints and attribute data
- Saves the data in the `data/atlas_data/` directory as:
  - `features_by_id.json` - Building attributes
  - `geometry_by_id.json` - Building geometries

### 2. Capture Satellite Images

Next, capture satellite imagery for the downloaded building data:

```bash
python 02_capture_images.py
```

This script:
- Uses Selenium to automate browser interaction with mapping services
- Captures satellite images for each building footprint
- Creates binary masks for building outlines
- Saves the results in `data/images/` as pairs of:
  - `image_[ID].png` - Satellite imagery
  - `mask_[ID].png` - Binary building mask

### 3. Generate AI Captions

Generate descriptive captions for each image using AI vision models:

```bash
python 03_extract_captions.py --model gpt-4o
```

This script:
- Processes each image in the dataset
- Sends images to the OpenAI API (via OpenRouter)
- Uses the system prompt from `config.yml` to guide caption generation
- Focuses on building characteristics relevant to heat demand
- Saves captions to a JSON file

### 4. Extract Embeddings

Convert captions into numerical embeddings for machine learning:

```bash
python 04_extract_embeddings.py
```

This script:
- Uses the Sentence Transformers library
- Creates 512-dimensional embeddings from each caption
- Normalizes the embeddings
- Combines them with geometrical features
- Saves the embeddings dataset for model training

### 5. Train Prediction Models

Train machine learning models to predict heat demand:

```bash
python 05_train.py
```

This script:
- Loads the embeddings and target data
- Splits data into training and validation sets
- Trains multiple regression models:
  - Gradient Boosting
  - Random Forest
  - Linear Regression
- Evaluates models using cross-validation
- Saves the best performing model to `models/best_gpt4o_regressor.joblib`

### 6. Run Model Ablations and CNN

Compare different model architectures and feature sets:

```bash
python 06_run_ablations.py
python 06_run_CNN.py
```

This script:
- Tests different combinations of features
- Compares text embeddings vs. CNN features
- Evaluates the contribution of geometric features
- Generates comparison charts and metrics

### 7. Interactive Demo Application

Once the models are trained, run the demo application:

```bash
python app.py
```

This launches a FastAPI web application that:
- Provides an interactive satellite map interface
- Allows users to draw polygons around building areas
- Calculates geometric features (area, perimeter)
- Counts buildings using the Overpass API
- Generates AI captions for the selected area
- Predicts heat demand using the trained model
- Displays results in a user-friendly interface

## Installation

### Prerequisites
- Python 3.9+
- Virtual environment
- Chrome/Firefox for Selenium (with appropriate webdriver)
- OpenRouter API key (for accessing GPT-4o)

### Setup

1. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```
OPENROUTER_API_KEY=your_openrouter_api_key
```
### Pretrained 

- The experimentation is conducted within the defined bounding box coordinates: 7.711388,49.410640,7.812840,49.468572.
- Download the pre-trained model from [Here](https://drive.google.com/drive/folders/17MeutRkhPiHjks8zz1jKkYKucLvXXaPp) and paste it under data/models.


## Project Structure

- Data processing scripts:
  - `01_fetch_data_bbox.py` - Download data from RLP Atlas
  - `02_capture_images.py` - Generate satellite images and masks
  - `03_extract_captions.py` - Generate AI captions
  - `04_extract_embeddings.py` - Create text embeddings
  - `05_train.py` - Train prediction models
  - `06_run_ablations.py` - Compare model variations

- Demo application:
  - `app.py` - FastAPI application server
  - `config.yml` - System prompts and configuration
  - `templates/index.html` - Web interface
  - `models/` - Trained model files

- Data directories:
  - `data/atlas_data/` - Raw data from RLP Atlas
  - `data/images/` - Satellite images and masks
  - `data/embeddings/` - Generated text embeddings
  - `data/models/` - Trained models and results

## API Endpoints

- `GET /`: Main application page
- `GET /health`: Health check endpoint
- `POST /upload-image`: Processes images and returns analysis

## Technical Implementation Details

### Heat Demand Prediction Model

The machine learning models predict heat demand using a combination of:

1. **Text Embeddings (512 dimensions)**
   - Generated from AI captions using Sentence Transformers
   - Normalized and layer-normalized for stability
   - Captures semantic building characteristics from visual data

2. **Geometric Features**
   - Area (square meters) of the polygon
   - Perimeter length (meters) of the polygon
   - Building count retrieved via Overpass API

3. **Model Architecture Options**
   - Gradient Boosting Regressor (default, best performance)
   - Random Forest Regression (alternative)
   - Linear Regression (baseline)
   - CNN features extraction (ablation study)

The embedding pipeline transforms qualitative descriptions into quantitative features that correlate with heat demand.

### System Prompt Engineering

The `config.yml` file contains a carefully engineered system prompt that guides the AI model to focus on heat-relevant building characteristics:

```yaml
system:
  prompt: |
    You are a municipal heat planner analyzing an aerial satellite image...
    Focus on observable urban characteristics, such as:
    Building density (e.g., number of buildings)
    Spacing between buildings (e.g., average distance in meters)
    ...
```

This prompt engineering is critical for obtaining consistent, relevant captions that correlate with heat demand.

## Acknowledgments

- RLP Atlas for providing building data
- ESRI for the ArcGIS JavaScript API
- OpenAI for the GPT-4o model
- OpenRouter for API access
- SentenceTransformers for text embedding capabilities
- The scikit-learn team for machine learning implementations
