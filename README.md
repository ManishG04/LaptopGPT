# LaptopGPT

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-3.1-green.svg)](https://flask.palletsprojects.com/)

> An intelligent recommendation system powered by GPT-4 and rule-based filtering for personalized laptop suggestions.

## Overview

LaptopGPT is a sophisticated recommendation system that leverages artificial intelligence to help users find the ideal laptop based on their specific requirements. By combining GPT-4's natural language understanding capabilities with a precise rule-based filtering system, it provides accurate and personalized laptop recommendations.

## Features

- **Natural Language Processing**: Advanced GPT-4 integration through LangChain for intuitive user interactions
- **Intelligent Filtering System**: Rule-based engine for precise laptop matching
- **RESTful API**: Flask-based backend for reliable communication
- **Interactive Interface**: Modern web interface with real-time chat capabilities
- **Data-Driven**: Utilizes comprehensive laptop dataset with detailed specifications
- **Smart Constraints**: Handles budget and requirements constraints efficiently
- **Use Case Optimization**: Specialized filtering for different user profiles (gaming, professional, academic)

## Requirements

- Python 3.12+
- OpenAI API key
- Poetry (package manager)
- Modern web browser

## Technology Stack

### Core Technologies

- **Backend Framework**: Flask 3.1.0
- **AI Integration**: 
  - OpenAI GPT-4
  - LangChain Framework
- **Database**: CSV-based dataset with efficient filtering
- **Frontend**: 
  - HTML5/CSS3
  - Vanilla JavaScript
  - Markdown rendering support
- **Development Tools**:
  - Poetry for dependency management
  - Pytest for testing

## Installation

1. Clone the repository
```bash
git clone https://github.com/ManishG04/LaptopGPT.git
cd LaptopGPT
```

2. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

3. Install dependencies
```bash
poetry install
```

4. Run the application
```bash
poetry run python app.py
```

## Architecture

The system follows a modular architecture:

```
User Request ─────┐
                 ▼
           Flask API Server
                 │
        ┌────────┴────────┐
        ▼                 ▼
    LangChain        Rule Engine
    (GPT-4)              │
        │                ▼
        │           Laptop Dataset
        │                │
        └────────┬───────┘
                 │
            Response
```

### Components

1. **API Layer** (`app.py`):
   - Handles HTTP requests
   - Manages session state
   - Routes communications

2. **AI Processing** (`chatbot.py`):
   - Integrates with OpenAI GPT-4
   - Manages conversation context
   - Processes natural language

3. **Recommendation Engine** (`recommendation.py`):
   - Implements filtering logic
   - Applies business rules
   - Ranks recommendations

4. **Data Layer** (`data/`):
   - Maintains laptop dataset
   - Handles data preprocessing
   - Manages data updates

## API Documentation

### Endpoints

#### POST /chat
Process user messages and return laptop recommendations.

```json
{
    "message": "I need a gaming laptop under $1000"
}
```

Response:
```json
{
    "response": "Based on your requirements, here are the recommended laptops..."
}
```


