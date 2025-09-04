# YouTube Thumbnail Effectiveness Predictor

This project analyzes YouTube thumbnails and predicts their effectiveness using machine learning techniques. It extracts visual features, metadata, and engagement metrics to identify patterns that contribute to thumbnail success.

## Project Structure
```
youtube_thumbnail/
├── data/                    # Data storage
│   ├── raw/                # Raw thumbnails and metadata
│   └── processed/          # Processed features and datasets
├── src/
│   ├── data/
│   │   ├── collector.py    # YouTube data collection
│   │   └── processor.py    # Data preprocessing
│   ├── features/
│   │   ├── visual.py       # Visual feature extraction
│   │   ├── composition.py  # Thumbnail composition analysis
│   │   └── text.py        # Title feature extraction
│   ├── models/
│   │   ├── trainer.py      # Model training
│   │   └── evaluator.py    # Model evaluation
│   └── utils/
│       └── helpers.py      # Utility functions
├── notebooks/              # Analysis notebooks
├── reports/                # Generated analysis reports
├── requirements.txt        # Project dependencies
└── config.py              # Configuration settings
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up YouTube API credentials:
- Create a project in Google Cloud Console
- Enable YouTube Data API v3
- Create API credentials
- Save the API key in `.env` file:
```
YOUTUBE_API_KEY=your_api_key_here
```

## Usage

1. Data Collection:
```bash
python src/data/collector.py
```

2. Feature Extraction:
```bash
python src/features/visual.py
python src/features/composition.py
python src/features/text.py
```

3. Model Training:
```bash
python src/models/trainer.py
```

4. Generate Insights:
```bash
python src/models/evaluator.py
```

## Success Criteria
- Model achieves >0.6 R² on held-out test set
- Clear identification of statistically significant visual patterns
- Practical recommendations for thumbnail optimization
