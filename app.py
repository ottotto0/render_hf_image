import os
import time
import base64
from flask import Flask, render_template, request, jsonify
from gradio_client import Client

app = Flask(__name__)

# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN")
SPACE_ID = "Nech-C/waiNSFWIllustrious_v140"

# Initialize Client
# Try to initialize with token if available
print(f"Initializing Gradio Client for {SPACE_ID}...")
try:
    if HF_TOKEN:
        print("Using HF_TOKEN for authentication.")
        client = Client(SPACE_ID, hf_token=HF_TOKEN)
    else:
        print("No HF_TOKEN found, using anonymous access.")
        client = Client(SPACE_ID)
except TypeError as e:
    print(f"Failed to init with hf_token: {e}")
    print("Falling back to anonymous access (or relying on env var if supported internally).")
    client = Client(SPACE_ID)
except Exception as e:
    print(f"Error initializing client: {e}")
    client = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if not client:
        return jsonify({'error': 'Backend not initialized'}), 500

    data = request.json
    prompt = data.get('prompt', '')
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    print(f"Generating image for prompt: {prompt}")
    
    try:
        # Call the API
        # predict(model, prompt, quality_prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps, num_images, use_quality, api_name="/infer")
        # We use defaults for most things
        result = client.predict(
            "v140",             # model
            prompt,             # prompt
            "masterpiece, best quality, fine details", # quality_prompt
            "blurry, low quality, watermark, monochrome, text", # negative_prompt
            0,                  # seed
            True,               # randomize_seed
            1024,               # width
            1024,               # height
            6,                  # guidance_scale
            30,                 # num_inference_steps
            1,                  # num_images
            True,               # use_quality
            api_name="/infer"
        )
        
        # Result structure: (result_image, result_original_size, seed, history)
        # result_image is a dict or filepath depending on version/config
        # In the API info it says: [Image] result: dict(path: str | None, url: str | None, ...)
        
        img_data = result[0]
        
        image_path = None
        if isinstance(img_data, dict):
            image_path = img_data.get('path') or img_data.get('url')
        elif isinstance(img_data, str):
            image_path = img_data
            
        if not image_path:
            return jsonify({'error': 'No image path returned'}), 500
            
        # If it's a local file, read it and convert to base64
        if os.path.exists(image_path):
            with open(image_path, "rb") as img_file:
                b64_string = base64.b64encode(img_file.read()).decode('utf-8')
                # Determine mime type (guess png or jpg)
                mime_type = "image/png" # Default
                if image_path.endswith('.jpg') or image_path.endswith('.jpeg'):
                    mime_type = "image/jpeg"
                elif image_path.endswith('.webp'):
                    mime_type = "image/webp"
                
                return jsonify({'image_url': f"data:{mime_type};base64,{b64_string}"})
        else:
            # It might be a URL
            return jsonify({'image_url': image_path})

    except Exception as e:
        print(f"Generation failed: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
