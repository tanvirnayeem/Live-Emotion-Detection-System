# haar cascade for face detection
detect_frontal_face = 'haarcascades/haarcascade_frontalface_alt.xml'
# emotion detection model
path_model = './Modelos/model_dropout.hdf5'
# Model parameters, the image should be converted to a 48x48 grayscale image
w,h = 48,48
rgb = False
labels = ['angry','disgust','fear','happy','neutral','sad','surprise']
