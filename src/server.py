from flask import Flask, request, render_template
from predict import predict
from util import downloadImages, removeFiles, downloadImage, removeFile, getBase64, getRandomStr, IMAGE_PATH
import os, io
from flask_cors import CORS
from flask import send_file, jsonify

app = Flask(__name__)
cors = CORS(app, resources={
  r"/eval/*": {"origin": "*"},
})

@app.route('/', methods=['GET'])
def main():
    return app.send_static_file('index.html')

@app.route('/healthz', methods=['GET'])
def healthz():
   return 'ok'

@app.route('/predict/url', methods=['POST'])
def url():
    localFilePath = ''
    output_file_path = ''
    try:
        url = request.json["source_url"]
        modelName = request.json["modelName"]
        print(modelName, flush=True)
        if modelName not in ['psnr-large', 'psnr-small', 'gans', 'noise-cancel']:
            return jsonify({'message': 'invalid params'}), 400

        for key in request.args:
            if key != 'url':
                url += '&' + key + '=' + request.args[key]
        localFilePath = downloadImage(url)
        output_file_path = predict(localFilePath, modelName)
        callback = send_file(output_file_path, mimetype='image/jpeg')
        return callback, 200
    except Exception as e:
        print(str(e), flush=True)
        return {'error': str(e)}
    finally:
        if (localFilePath != ''):
            removeFile(localFilePath)
        if (localFilePath != ''):
            removeFile(output_file_path)

@app.route('/predict/file', methods=['POST'])
def file():
    localFilePath = ''
    output_file_path = ''
    try:
        file = request.files['file']
        modelName = request.form['modelName']
        if modelName not in ['psnr-large', 'psnr-small', 'gans', 'noise-cancel']:
            return jsonify({'message': 'invalid params'}), 400
        if not file:
            return jsonify({'message': 'nofile'}), 400
        if file.content_type not in ['image/jpeg', 'image/jpg', 'image/png']:
            return jsonify({'message': 'only support jpeg, jpg or png'}), 400
        localFilePath =  IMAGE_PATH + '/' + getRandomStr(15) + '.jpg'
        file.save(localFilePath)
        output_file_path = predict(localFilePath, modelName)
        callback = send_file(output_file_path, mimetype='image/jpeg')
        return callback, 200
    except Exception as e:
        print(str(e), flush=True)
        return {'error': str(e)}
    finally:
        if (localFilePath != ''):
            removeFile(localFilePath)
        if (localFilePath != ''):
            removeFile(output_file_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)