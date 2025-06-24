import sys
import os
from flask import Flask, jsonify, send_from_directory
from bson import json_util
import json

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.database.mongo_db import MongoDB

# --- Flask App Initialization ---
app = Flask(__name__, static_folder='static', static_url_path='')

# --- Database Connection ---
db_handler = MongoDB()
signals_collection = db_handler.get_collection('signals')

# --- API Endpoints ---
@app.route('/api/signals', methods=['GET'])
def get_signals():
    """API endpoint to fetch all signals from the database."""
    try:
        all_signals = list(signals_collection.find({}, {'_id': 0})) # Exclude the default '_id'
        return jsonify(all_signals)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    """Serves the main HTML page."""
    return send_from_directory(app.static_folder, 'index.html')

# --- Main Execution ---
if __name__ == '__main__':
    # The static folder should be at the root of the project for simplicity
    # This allows 'index.html' to be served from 'src/api/static'
    app.static_folder = 'static'
    app.run(debug=True, port=5000) 