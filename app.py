import numpy as np
import streamlit as st
import pickle 
from keras.utils import load_img, img_to_array
from skimage.feature import hog
from skimage.feature import local_binary_pattern as LBP
from skimage.feature import hog, orb, sift
import seaborn as sns
from sklearn.svm import SVC

# List of model filenames
model_filenames = ['hogLogisticRegressionModel.pkl', 'hogSVMModel.pkl', 'LbpLogisticRegressionModel.pkl', 'LbpSVMModel.pkl', 'logisticRegressionModel.pkl', 'svmModel.pkl']

# Dictionary to store loaded models
loaded_models = {}

# Load each model
for filename in model_filenames:
    with open(filename, 'rb') as m:
        loaded_models[filename] = pickle.load(m)
# st.write(loaded_models['hogSVMModel.pkl'])
        
LogisticModel ={
    'Select a Sub Model' : None,
    'LogisticRegressionModel' : loaded_models['logisticRegressionModel.pkl'],
    'LbpLogisticRegressionModel' : loaded_models['LbpLogisticRegressionModel.pkl'],
    'hogLogisticRegressionModel' : loaded_models['hogLogisticRegressionModel.pkl']
}
svm = {
    'Select a Sub Model' : None,
    'svmModel' : loaded_models['svmModel.pkl'],
    'LbpSVMModel' : loaded_models['LbpSVMModel.pkl'],
    'hogSVMModel' : loaded_models['hogSVMModel.pkl']
}

mainModel = ['select a Model','LogisticModel', 'SVMmodel']

classes = ['car', 'cricket Ball', 'Ice Cream Cone']
st.title('Welcome Amin!')
st.title('Image Classification APP')
uploader = st.file_uploader('Select Image', type=['jpg', 'jpeg', 'png'])
if uploader is not None:
    IMG = load_img(uploader)
    st.image(IMG)
    img = load_img(uploader, target_size=(100,100), color_mode='grayscale')
    imgArr = img_to_array(img)
    # print(imgArr.squeeze(2).shape) 
    # img2D = imgArr.squeeze(2)
    selected_model = st.selectbox('Select a Model', mainModel)


    if selected_model == 'LogisticModel':
        selected_model_name = st.selectbox('Choose a Model', list(LogisticModel.keys()))
        if selected_model_name  =='LogisticRegressionModel':
            imgFlatten= imgArr.flatten()
            st.write('flatten shape is', imgFlatten.shape)
            res = loaded_models['logisticRegressionModel.pkl'].predict([imgFlatten])
            st.write(f'Model Predicted Class is {classes[res[0]]}')
            st.image(img)
            st.write(loaded_models['logisticRegressionModel.pkl'])

        elif selected_model_name  =='LbpLogisticRegressionModel':
            lbpFeat = LBP(imgArr[:, :, 0], 3, 1).flatten()
            res = loaded_models['LbpLogisticRegressionModel.pkl'].predict([lbpFeat])
            st.write(f'Model Predicted Class is {classes[res[0]]}')
            st.image(img)


        elif selected_model_name  =='hogLogisticRegressionModel':
            img2D = imgArr.squeeze(2)
            hogfeat, _ = hog(img2D, visualize=True)
            hogFeat = hogfeat.flatten()
            # Assuming loaded_models is defined somewhere in your code
            res = loaded_models['hogLogisticRegressionModel.pkl'].predict([hogFeat])
            st.write(f'Model Predicted Class is {classes[res[0]]}')
            st.image(img)

    if selected_model == 'SVMmodel':
        selected_model_name = st.selectbox('Choose a Model', list(svm.keys()))
        if selected_model_name  =='svmModel':
            imgFlatten= imgArr.flatten()
            st.write('flatten shape is', imgFlatten.shape)
            res = loaded_models['svmModel.pkl'].predict([imgFlatten])
            st.write(f'Model Predicted Class is {classes[res[0]]}')
            st.image(img)
            st.write(loaded_models['svmModel.pkl'])

        elif selected_model_name  =='LbpSVMModel':
            lbpFeat = LBP(imgArr[:, :, 0], 3, 1).flatten()
            res = loaded_models['LbpSVMModel.pkl'].predict([lbpFeat])
            st.write(f'Model Predicted Class is {classes[res[0]]}')
            st.image(img)
            
        elif selected_model_name  =='hogSVMModel':
            img2D = imgArr.squeeze(2)
            hogfeat, _ = hog(img2D, visualize=True)
            hogFeat = hogfeat.flatten()
            # Assuming loaded_models is defined somewhere in your code
            res = loaded_models['hogSVMModel.pkl'].predict([hogFeat])
            st.write(f'Model Predicted Class is {classes[res[0]]}')
            st.image(img)
            


