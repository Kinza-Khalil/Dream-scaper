from flask import Flask, request, send_file, render_template, jsonify
import os
import requests
import base64
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

api_host = 'https://api.stability.ai'
api_key = os.environ.get("API_KEY")
engine_id = 'stable-diffusion-xl-beta-v2-2-2'

@app.route("/")
@cross_origin()
def home():
    return render_template('index.html')

@app.route('/static/<path:path>')
@cross_origin()
def static_file(path):
    try:
        return app.send_static_file(path)
    except:
        return jsonify({'error': f"Could not find file: {path}"}), 404

@app.route("/getModels", methods=['GET'])
@cross_origin()
def getModelList():
    url = f"{api_host}/v1/engines/list"
    try:
        response = requests.get(url, headers={"Authorization": f"Bearer {api_key}"})
        if response.status_code == 200:
            return response.json()
        else:
            return jsonify({'error': 'Unable to fetch models'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/generateImage", methods=['POST'])
@cross_origin()
def generateStableDiffusionImage():
    try:
        prompt = request.json['prompt']
        height = request.json.get('height', 512)
        width = request.json.get('width', 512)
        steps = request.json.get('steps', 50)

        url = f"{api_host}/v1/generation/{engine_id}/text-to-image"
        headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}"
        }
        payload = {}
        payload['text_prompts'] = [{"text": f"{prompt}"}]
        payload['cfg_scale'] = 7
        payload['clip_guidance_preset'] = 'FAST_BLUE'
        payload['height'] = height
        payload['width'] = width
        payload['samples'] = 1
        payload['steps'] = steps

        response = requests.post(url,headers=headers,json=payload)

        if response.status_code == 200:
            data = response.json()
            filename = f"v1_txt2img.png"
            for i, image in enumerate(data["artifacts"]):
                with open(filename, "wb") as f:
                    f.write(base64.b64decode(image["base64"]))
            return send_file(filename, mimetype='image/png')
        else:
            return jsonify({'error': 'Unable to generate image'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
