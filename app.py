import re
from flask import Flask, app, render_template,request
import numpy as np
import os
#from tensorflow.keras.models import load_model
from keras.models import load_model
#from tensorflow.keras.preprocessing import image
from keras.preprocessing import image
#from tensorflow.python.keras.backend import argmax
from keras.backend import argmax
from werkzeug.utils import secure_filename
import tensorflow_addons as tfa
import glob


# Flask app
app = Flask(__name__)


model = load_model('models/Final_apple_leaf_desease_detector_ResNet50_V2.h5')






def model_predict(img_path,model):
    test_image= image.load_img(img_path,target_size=(256,256))
    test_image= image.img_to_array(test_image)
    test_image= test_image/255
    test_image=np.expand_dims(test_image,axis=0) # to tell NN Dim is (1,256,256,3)
    result=model.predict(test_image)
    result = result.tolist()[0]
    return result





#deleting files
def delete_files():
    #Upload folder path
    files = glob.glob('uploads/*')
    for f in files:
        if len(files)>3:
            os.remove(f)






@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        
        basepath=os.path.dirname(os.path.realpath('__file__'))
        file_path=os.path.join(basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)
   
    # Make Predictions
    Labels = ['Complex', 'Frog eye', 'Healthy', 'Powdery Mildew', 'Rust', 'Scab']
    result = model_predict(img_path=file_path,model=model)
    predictions =[]
    for i in result:
        if i>=0.4:
            predictions.append(Labels[result.index(i)])
    #deleting files in upload
    delete_file = delete_files()        
    return str(predictions)        
    
  

if __name__=='__main__':
    app.run(debug=False,port=5000)    




