# VNStock Data Explorer

A minimal Flask web app that connects to the **vnstock** library and displays sample stock data on the homepage.

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Start the development server:

   ```bash
   python app.py
   ```

The home page fetches data for ticker `ACB` using the `vnstock` package. Errors are displayed on the page if the data source is unavailable.
