from flask import Flask, request, jsonify
from flask_cors import CORS
from server.seach import retrieve_images
import logging
app = Flask(__name__)
CORS(app)  # 允许所有域的跨域请求

@app.route('/retrieve_images', methods=['POST'])
def retrieve_images_endpoint():
    data = request.json
    query_text = data['query']
    top_k_predictions = retrieve_images(query_text)
    return jsonify({'results': top_k_predictions})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    app.run(debug=True)
