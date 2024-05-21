from flask import Flask, session, request, jsonify
import sys
sys.path.append('../LatexOCR')
from preprocessor import Preprocessor
import logging
import numpy as np
from io import BytesIO
from PIL import Image

app = Flask(__name__)
preprocessor = Preprocessor()

logger = logging.Logger(__name__)


@app.route('/', methods=['POST'])
def index():
    if request.method == 'POST':
        # data = request.json
        logger.info(f'Got data: {request.form}')
        # print('Got data:', request.files['image'])
        # data = request.form.get('image')
        data = request.files['image'].read()
        # print(f'data type: {type(data)}')
        # arr = np.frombuffer(data)
        # image = np.array(data['image'])

        # logger.info(f'Got image with shape: {image.shape}, type: {type(image)}')

        bytes_stream = BytesIO(data)
        image = Image.open(bytes_stream)
        image = np.array(image)

        print(f'Got image with shape: {image.shape}, type: {type(image)}')

        threshold = int(request.form.get('threshold', 0))
        
        # threshold = data['threshold']
        print(f'Got threshold: {threshold}, type: {type(threshold)}')

        # apply_fft = data['apply_fft']
        

        apply_fft = request.form.get('apply_fft', False)
        if apply_fft == 'True':
            apply_fft = True
        elif apply_fft == 'False':
            apply_fft = False

        print(f'Got apply_fft status: {apply_fft}, type: {type(apply_fft)}')

        if not threshold:
            logger.warning(f'process_image_automatically method')
            new_image, threshold = preprocessor.process_image_automatically(image, apply_fft=apply_fft, percentile=70)
        
        else:
            logger.warning(f'process_image method')
            new_image = preprocessor.process_image(image, apply_fft=apply_fft, contrast_threshold=threshold)
    
    return jsonify(image=str(new_image.tobytes()), threshold=threshold)
