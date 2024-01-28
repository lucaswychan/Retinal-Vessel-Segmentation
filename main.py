from utils import *
from model import *
from data_process import *
import numpy as np
import keras

def main():
    retinal_vessel_data = np.load('Data/retinal_vessel_dataset.npz')
    implementation_check = np.load('Data/sample_data.npz')
    
    x_train_raw = retinal_vessel_data["x_train"][...,np.newaxis]
    y_train = retinal_vessel_data["y_train"][...,np.newaxis].astype(int)
    
    x_val_raw = retinal_vessel_data["x_val"][...,np.newaxis]
    y_val = retinal_vessel_data["y_val"][...,np.newaxis].astype(int)
    
    x_train_enhanced = contrast_stretch(x_train_raw)
    x_val_enhanced = contrast_stretch(x_val_raw)
    
    x_train = rescale_01(x_train_enhanced)
    x_val = rescale_01(x_val_enhanced)
    keras.utils.set_random_seed(1016) # Reset seed, you should get the same model since this code cell

    learning_rate = 0.001  # Feel free to experiment with different values!
    num_epochs = 100       # Feel free to experiment with different values!

    model = build_model() # Build new model with newly initialized weights
    model.summary()
    compile_model(model, learning_rate)
    train_model(model, num_epochs, x_train, y_train, x_val, y_val)

    val_preds = predict_model(model, x_val)
    
    visualize_side_by_3(x_val[12,...], 'image', y_val[12,...], 'label', val_preds[12,...], 'predicted (before threshold)', "12_image_before_thresh", (0,1),(0,1),(0,1))
    visualize_side_by_3(x_val[112,...], 'image', y_val[112,...], 'label', val_preds[112,...], 'predicted (before threshold)', "112_image_before_thresh", (0,1),(0,1),(0,1))
    
    val_preds_thresh = threshold(val_preds, 0.5)  

    visualize_side_by_3(x_val[12,...], 'image', y_val[12,...], 'label', val_preds_thresh[12,...], 'predicted (after threshold)', "12_image_after_thresh", (0,1),(0,1),(0,1))
    visualize_side_by_3(x_val[112,...], 'image', y_val[112,...], 'label', val_preds_thresh[112,...], 'predicted (after threshold)', "112_image_after_thresh", (0,1),(0,1),(0,1))
    
    sample_img = implementation_check['sample_img'] 
    sample_label = implementation_check['sample_label']
    sample_pred = implementation_check['sample_pred_hard']

    visualize_side_by_3(sample_img, 'image', sample_label, 'label', sample_pred, 'predicted', "sample_image", (0,255),(0,1),(0,1))
    print("Dice coefficient: {:.4f}".format(dice_coef(sample_label, sample_pred)))
    
    print("{:.4f}".format(avg_dice(y_val, val_preds_thresh)))
    
    model.save('Model/trained_model_snapshot.h5')
        
        
if __name__ == '__main__':
    main()