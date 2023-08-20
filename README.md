# Flipkart_GRiD_Information_Security
# Log Analysis and Conversational AI Project

This project aims to analyze log data and implement conversational AI using Language Models (LLMs) for enhancing interaction with multiple document types. It includes log analysis, text processing, and integration with Streamlit for interactive conversation.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Running Log Analysis](#running-log-analysis)
- [Running Conversational AI](#running-conversational-ai)
- [HTML Templates](#html-templates)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project combines log analysis and conversational AI to enhance security and document interaction. The log analysis component parses server logs, identifies failed attempts, and summarizes the results in a CSV file. The conversational AI component utilizes Language Models to process and interact with PDFs, CSVs, and TXT documents.

## Installation

To use this project, follow these steps:

1. Clone the repository:

git clone https://github.com/yourusername/log-analysis-conversational-ai.git
cd log-analysis-conversational-ai


2. Set up your development environment:
```
pip install -r requirements.txt
```

## Dependencies
The project requires the following dependencies:

streamlit
pypdf2
langchain
python-dotenv
openai
faiss-cpu (for vectorstore)
altair (for visualization)
tiktoken

## Usage
Running Log Analysis
The main.py script performs log analysis and generates a CSV file summarizing failed and successful attempts. Follow these steps:

Ensure your server logs are in the same directory as serverlogs.log.
Run the script:
```
python main.py
```

Running Conversational AI
The app.py script integrates with Streamlit to create an interactive conversational AI experience for PDFs, CSVs, and TXT documents. Follow these steps:

Upload your documents (PDF, CSV, TXT) through the Streamlit interface.
Ask questions related to the uploaded documents.
Observe the AI's responses and interaction history.
HTML Templates
The htmlTemplate.py file contains HTML templates for styling chat messages. It defines styles for user and bot messages in the conversation.

## Requirements
The requirements.txt file lists the required dependencies to run the project. Install the dependencies using:

```
pip install -r requirements.txt
```



