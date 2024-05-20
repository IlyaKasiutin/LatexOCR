import json

from flask import Flask, session, request, jsonify
import sys
sys.path.append('../LatexOCR')
from preprocessor import Preprocessor
from io import BytesIO
from PIL import Image
import logging
import numpy as np

app = Flask(__name__)
preprocessor = Preprocessor()

logger = logging.Logger(__name__)


@app.route('/', methods=['POST'])
def index():
    if request.method == 'POST':
        data = request.json
        # data = request.form.get('image')
        # arr = np.frombuffer(data)
        image = np.array(data['image'])

        logger.info(f'Got image with shape: {image.shape}, type: {type(image)}')

        # bytes_stream = BytesIO(data)
        # image = Image.open(bytes_stream)

        # threshold = request.form.get('threshlod')
        # apply_fft = request.form.get('apply_fft')
        threshold = data['threshold']
        logger.info(f'Got threshold: {threshold}, type: {type(threshold)}')

        apply_fft = data['apply_fft']
        logger.info(f'Got apply_fft status: {apply_fft}, type: {type(apply_fft)}')

        if not threshold:
            logger.warning(f'process_image_automatically method')
            new_image, threshold = preprocessor.process_image_automatically(image, apply_fft=apply_fft, percentile=70)
        
        else:
            logger.warning(f'process_image method')
            new_image = preprocessor.process_image(image, apply_fft=apply_fft, contrast_threshold=threshold)
                
    return jsonify(image=new_image.tolist(), threshold=threshold)



if __name__ == '__main__':
    app.run(host='127.0.0.1', port=7003)
