# YoloCropper
QT Python cropping tool that uses the Yolo8 AI model for face an body cropping

This uses the Yolo8 model for detecting body or face and gives you the option to crop images and automatically keep the body and head in the cropped photo.
I created it because I needed to crop 1000's of training images to train Stable Diffusion SDXL models using Kohya SS Gui
Because this is intended for Stable Diffusion SDXL it will create an additional 1024x1024 or 1024x1536 version of each image depending on which ratio is chosen. 

ACQUIRING THE MODELS

For the main body detection you will need to download the main models from here
https://github.com/ultralytics/ultralytics

Pick one or all of the following models
YOLOv8n,YOLOv8s,YOLOv8m,YOLOv8l,YOLOv8x

Then you will need a face trained model which you can get from here

https://github.com/akanametov/yolov8-face
or here
https://huggingface.co/models?sort=trending&search=yolo+face

INSTALL PYTORCH FROM HERE

https://pytorch.org/get-started/locally/

I used.... pip3 install torch torchvision torchaudio

INSTALL YOLO FROM HERE

https://docs.ultralytics.com/quickstart/

USING THE CODE

Either run main.py using python main.py

or

Create a new QT Creator Python project
Overwrite the code in main.py with the code from this main.py
Ensure you use pip to install the appropriate python modules
to do this go into your project folder and type source venv/bin/activate
e.g pip install PyQt5 opencv-python numpy ultralytics

