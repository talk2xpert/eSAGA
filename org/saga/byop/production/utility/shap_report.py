import shap
def f(x):
    tmp = x.copy()
    return model(tmp)
masker_blur = shap.maskers.Image("blur(32,32)", x_test_mstar[0].shape)
explainer = shap.Explainer(f, masker_blur, output_names=list(range(3)))

ind=[600]
shap_values_ = explainer( x_test_mstar[ind], max_evals=5000, batch_size=50 )
shap.image_plot(shap_values_,labels=[0,1,2])


background = x_train_mstar[np.random.choice(x_train_mstar.shape[0], 100, replace=False)]
e = shap.DeepExplainer(model, background)
ind=[600]
shap_values = e.shap_values( x_test_mstar[ind])
shap.image_plot(shap_values,x_test_mstar[ind])



####https://github.com/amirhoseinoveis/SHAP-with-MSTAR/blob/main/SHAP-with-MSTAR.ipynb

x = Conv2D(32, 3, activation="relu", padding='same',name='Conv1')(inputs)
x = MaxPooling2D(2,name='Pool1')(x)
x = Conv2D(16, 3, activation="relu",padding='same',name='Conv2')(x)
x = MaxPooling2D(2,name='Pool2')(x)
x = Flatten(name='Vectorize')(x)
x= Dense(3,name='FC')(x)