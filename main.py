import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, QLabel, QTextEdit, QCheckBox,QProgressBar,QHBoxLayout,QFrame,QSpacerItem,QSizePolicy,QGridLayout,QRadioButton,QButtonGroup
from PyQt5.QtGui import QPixmap,QImage
from pathlib import Path
from PyQt5.QtCore import Qt
from ultralytics import YOLO
import cv2
import numpy as np

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_subfolders(source_folder):
    edited_folder = os.path.join(source_folder, 'edited')
    ensure_directory_exists(edited_folder)
    for subfolder in ['Body-1to1', 'Body-1to1.5', 'Face-1to1', 'Face-1to1.5', 'BodyCropped-AnyRatio','SubjectDetectionResults']:
        ensure_directory_exists(os.path.join(edited_folder, subfolder))

def adjust_and_scale_box(box, aspect_ratio, scale_factor, img_shape, aspect_ratios, head_priority_radio,head_priority_radio2, head_priority_radio3, head_priority_radio4,anchor_point=None):
    original_x1, original_y1, original_x2, original_y2 = map(int, box)
    original_width = original_x2 - original_x1
    original_height = original_y2 - original_y1
    h, w, _ = img_shape
    new_size = None  # Initialize new_size to None
    print(f"Before while true, anchor_point = {anchor_point}")
    while True:
        x1, y1, x2, y2 = original_x1, original_y1, original_x2, original_y2
        width, height = original_width, original_height

        if anchor_point:
            x_anchor, y_anchor = anchor_point

        if aspect_ratio == '1:1':
            new_size = min(width, height) * scale_factor
            if (head_priority_radio.isChecked() and aspect_ratios['body_1_1'][0]) or \
                (head_priority_radio3.isChecked() and aspect_ratios['face_1_1'][0]):

                anchor_point = (x_anchor, y_anchor)
                print("using anchor point")
                x1 = int(x_anchor - (new_size / 2))
                x2 = int(x_anchor + (new_size / 2))
                y1 = int(y_anchor)
                y2 = int(y_anchor + new_size)
            else:
                print("not using anchor point")
                x_center, y_center = x1 + width // 2, y1 + height // 2
                x1 = int(x_center - (new_size / 2))
                x2 = int(x_center + (new_size / 2))
                y1 = int(y_center - (new_size / 2))
                y2 = int(y_center + (new_size / 2))

        elif aspect_ratio == '1:1.5':
            new_width = min(width, height) * scale_factor
            new_height = int(new_width * 1.5)

            if (head_priority_radio2.isChecked() and aspect_ratios['body_1_1_5'][0]) or \
                (head_priority_radio4.isChecked() and aspect_ratios['face_1_1_5'][0]):
                x1 = int(x_anchor - (new_width / 2))
                x2 = int(x_anchor + (new_width / 2))
                y1 = int(y_anchor)
                y2 = int(y_anchor + new_height)
            else:
                x_center, y_center = x1 + width // 2, y1 + height // 2
                x1 = int(x_center - (new_width / 2))
                x2 = int(x_center + (new_width / 2))
                y1 = int(y_center - (new_height / 2))
                y2 = int(y_center + (new_height / 2))

            new_size = f"{new_width}x{new_height}"  # Set new_size as a string for 1:1.5 aspect ratio

        if x1 >= 0 and x2 <= w and y1 >= 0 and y2 <= h:
            break  # If the box fits within the image, break

        scale_factor = max(0.1, scale_factor - 0.01)  # Reduce scale factor but not below 0.1
        print(f"Scale Factor - {scale_factor}")

        if scale_factor == 0.1:
            break  # Stop the loop if scale_factor reaches 0.1

    print(f"Aspect Ratio: {aspect_ratio}, New size: {new_size}, Coords: {x1}, {y1}, {x2}, {y2}")
    return x1, y1, x2, y2, scale_factor  # Return the coordinates and the final scale factor


def detect_and_crop(body_model, face_model, source_folder, info_textbox, aspect_ratios, progress_bar, original_image_label, cropped_image_label, head_priority_radio, head_priority_radio2,head_priority_radio3, head_priority_radio4,conf_threshold=0.30):
    print(f"head_priority_radio type = {type(head_priority_radio)}, value = {head_priority_radio.isChecked() if hasattr(head_priority_radio, 'isChecked') else head_priority_radio}")
    print(f"head_priority_radio2 type = {type(head_priority_radio2)}, value = {head_priority_radio2.isChecked() if hasattr(head_priority_radio2, 'isChecked') else head_priority_radio2}")
    print(f"head_priority_radio3 type = {type(head_priority_radio3)}, value = {head_priority_radio3.isChecked() if hasattr(head_priority_radio3, 'isChecked') else head_priority_radio3}")
    print(f"head_priority_radio4 type = {type(head_priority_radio4)}, value = {head_priority_radio4.isChecked() if hasattr(head_priority_radio4, 'isChecked') else head_priority_radio4}")
    print(f"conf_threshold={conf_threshold}")


    try:
        image_extensions = ['.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp']
        create_subfolders(source_folder)
        edited_folder = os.path.join(source_folder, 'edited')

        info_textbox.append("Starting image processing...")
        info_textbox.append("------")
        QApplication.processEvents()
        total_images = len([name for name in os.listdir(source_folder) if name.lower().endswith(tuple(image_extensions))])
        processed_images = 0

        crop_img = None  # Initialize crop_img as None
        for img_name in os.listdir(source_folder):
            try:
                if 'edited' in img_name:
                    continue

                info_textbox.append(f"Processing Image: {img_name}")
                QApplication.processEvents()

                if Path(img_name).suffix.lower() in image_extensions:

                    # Update the processed_images counter and the progress bar
                    processed_images += 1
                    progress = int((processed_images / total_images) * 100)
                    progress_bar.setValue(progress)

                    img_path = os.path.join(source_folder, img_name)
                    results_body = body_model.predict(img_path)
                    results_face = face_model.predict(img_path)

                    boxes_body = results_body[0].boxes.xyxy
                    confidences_body = results_body[0].boxes.conf
                    labels_body = results_body[0].boxes.cls

                    boxes_face = results_face[0].boxes.xyxy
                    confidences_face = results_face[0].boxes.conf
                    labels_face = results_face[0].boxes.cls

                    img = cv2.imread(img_path)
                    face_detected = False
                    body_detected = False

                    if aspect_ratios.get('body_cropped_any', (False,))[0]:
                        print("Entered body_cropped_any block")
                        for box, conf, label in zip(boxes_body, confidences_body, labels_body):
                            if label == 0 and conf >= conf_threshold:
                                x1, y1, x2, y2 = map(int, box)
                                crop_img = img[y1:y2, x1:x2]
                                h, w, _ = crop_img.shape
                                if crop_img.size != 0:
                                    png_path = os.path.join(edited_folder, f"BodyCropped-AnyRatio/{img_name.split('.')[0]}_{w}x{h}.png")
                                    cv2.imwrite(png_path, crop_img, [cv2.IMWRITE_PNG_COMPRESSION, 4])

                                    info_textbox.append("  Image saved successfully for body with any ratio.")
                                    QApplication.processEvents()
                                    info_textbox.append(f"  Final Resolution: {w}x{h}")
                                    QApplication.processEvents()

                    if aspect_ratios.get('subject_detection_results', (False,))[0]:
                        print("Entered subject_detection_results block")
                        img_with_boxes = img.copy()
                        for box, conf, label in zip(boxes_body, confidences_body, labels_body):
                            if label == 0 and conf >= conf_threshold:
                                x1, y1, x2, y2 = map(int, box)
                                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        for box, conf, label in zip(boxes_face, confidences_face, labels_face):
                            if label == 0 and conf >= conf_threshold:
                                x1, y1, x2, y2 = map(int, box)
                                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                png_compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 4]
                                cv2.imwrite(f"{edited_folder}/SubjectDetectionResults/{img_name.split('.')[0]}.png", img_with_boxes, png_compression_params)

                    # Find anchor point from face detection
                    anchor_point = None
                    for box, conf, label in zip(boxes_face, confidences_face, labels_face):
                        if label == 0 and conf >= conf_threshold:
                            x1, y1, x2, y2 = map(int, box)
                            anchor_point = ((x1 + x2) // 2, y1)
                            break  # Stop after finding the first face box

                    # Existing code for detecting and saving body images
                    for box, conf, label in zip(boxes_body, confidences_body, labels_body):
                        if label == 0 and conf >= conf_threshold:
                            body_detected = True
                            info_textbox.append("  Body detected.")
                            QApplication.processEvents()

                            # Find anchor point from face detection for x coordinate
                            x_anchor = None
                            for box, conf, label in zip(boxes_face, confidences_face, labels_face):
                                if label == 0 and conf >= conf_threshold:
                                    x1, y1, x2, y2 = map(int, box)
                                    x_anchor = (x1 + x2) // 2
                                    break  # Stop after finding the first face box

                            # Get y coordinate from the top of the body bounding box
                            y_anchor = None
                            for box, conf, label in zip(boxes_body, confidences_body, labels_body):
                                if label == 0 and conf >= conf_threshold:
                                    x1, y1, x2, y2 = map(int, box)
                                    y_anchor = y1
                                    break  # Stop after finding the first body box

                            anchor_point = None
                            if x_anchor is not None and y_anchor is not None:
                                if (head_priority_radio.isChecked() and aspect_ratios['body_1_1'][0]) or \
                                   (head_priority_radio2.isChecked() and aspect_ratios['body_1_1_5'][0]) or \
                                   (head_priority_radio3.isChecked() and aspect_ratios['face_1_1'][0]) or \
                                   (head_priority_radio4.isChecked() and aspect_ratios['face_1_1_5'][0]):
                                    # Use the x value from the face and the y value from the body for the anchor point
                                    anchor_point = (x_anchor, y_anchor)
                                else:
                                    # Use the x and y value from the face for the anchor point
                                    anchor_point = ((x1 + x2) // 2, y1)

                            if aspect_ratios['body_1_1'][0]:
                                x1, y1, x2, y2, final_scale = adjust_and_scale_box(box, '1:1', 3.0, img.shape,aspect_ratios, head_priority_radio,head_priority_radio2, head_priority_radio3, head_priority_radio4,anchor_point=anchor_point)
                                info_textbox.append(f"  Applying scale factor {final_scale} for body...")
                                QApplication.processEvents()
                                crop_img = img[y1:y2, x1:x2]
                                h, w, _ = crop_img.shape
                                if crop_img.size != 0:

                                    cv2.imwrite(f"{edited_folder}/Body-1to1/{img_name.split('.')[0]}_{w}x{h}_scale_{final_scale:.2f}.png", crop_img, [cv2.IMWRITE_PNG_COMPRESSION, 4])

                                    # New code to resize and save as 1024x1024
                                    resized_img = cv2.resize(crop_img, (1024, 1024))
                                    cv2.imwrite(f"{edited_folder}/Body-1to1/{img_name.split('.')[0]}_1024x1024_scale_{final_scale:.2f}.png", resized_img, [cv2.IMWRITE_PNG_COMPRESSION, 4])

                                    info_textbox.append("  Image saved successfully for body.")
                                    QApplication.processEvents()
                                    info_textbox.append(f"  Final Resolution: {w}x{h}")
                                    QApplication.processEvents()

                            if aspect_ratios['body_1_1_5'][0]:

                                x1, y1, x2, y2, final_scale = adjust_and_scale_box(box, '1:1.5', 3.0, img.shape,aspect_ratios,head_priority_radio,head_priority_radio2, head_priority_radio3, head_priority_radio4, anchor_point=anchor_point)
                                info_textbox.append(f"  Applying scale factor {final_scale} for body (1:1.5)...")
                                QApplication.processEvents()
                                crop_img = img[y1:y2, x1:x2]
                                h, w, _ = crop_img.shape
                                if crop_img.size != 0:
                                    cv2.imwrite(f"{edited_folder}/Body-1to1.5/{img_name.split('.')[0]}_{w}x{h}_scale_{final_scale:.2f}.png", crop_img, [cv2.IMWRITE_PNG_COMPRESSION, 4])

                                    # New code to resize and save as 1024x1536
                                    resized_img = cv2.resize(crop_img, (1024, 1536))
                                    cv2.imwrite(f"{edited_folder}/Body-1to1.5/{img_name.split('.')[0]}_1024x1536_scale_{final_scale:.2f}.png", resized_img, [cv2.IMWRITE_PNG_COMPRESSION, 4])

                                    info_textbox.append("  Image saved successfully for body (1:1.5).")
                                    QApplication.processEvents()
                                    info_textbox.append(f"  Final Resolution: {w}x{h}")
                                    QApplication.processEvents()

                    if not body_detected:
                        info_textbox.append("  No body detected.")
                        QApplication.processEvents()

                    for box, conf, label in zip(boxes_face, confidences_face, labels_face):
                        if label == 0 and conf >= conf_threshold:
                            face_detected = True
                            info_textbox.append("  Face detected.")
                            QApplication.processEvents()

                            if aspect_ratios['face_1_1'][0]:
                                print ("aspect_ratio is face_1_1")
                                print(f"Right before calling adjust_and_scale_box, anchor_point = {anchor_point}")
                                x1, y1, x2, y2, final_scale = adjust_and_scale_box(box, '1:1', 2.0, img.shape,aspect_ratios,head_priority_radio,head_priority_radio2, head_priority_radio3, head_priority_radio4, anchor_point=anchor_point)
                                info_textbox.append(f"  Applying scale factor {final_scale} for face...")
                                QApplication.processEvents()
                                crop_img = img[y1:y2, x1:x2]
                                h, w, _ = crop_img.shape
                                if crop_img.size != 0:
                                    cv2.imwrite(f"{edited_folder}/Face-1to1/{img_name.split('.')[0]}_{w}x{h}_scale_{final_scale:.2f}.png", crop_img, [cv2.IMWRITE_PNG_COMPRESSION, 4])

                                    # New code to resize and save as 1024x1024
                                    resized_img = cv2.resize(crop_img, (1024, 1024))
                                    cv2.imwrite(f"{edited_folder}/Face-1to1/{img_name.split('.')[0]}_1024x1024_scale_{final_scale:.2f}.png", resized_img, [cv2.IMWRITE_PNG_COMPRESSION, 4])

                                    info_textbox.append("  Image saved successfully for face.")
                                    QApplication.processEvents()
                                    info_textbox.append(f"  Final Resolution: {w}x{h}")
                                    QApplication.processEvents()

                            if aspect_ratios['face_1_1_5'][0]:
                                x1, y1, x2, y2, final_scale = adjust_and_scale_box(box, '1:1.5', 2.0, img.shape,aspect_ratios,head_priority_radio,head_priority_radio2, head_priority_radio3, head_priority_radio4, anchor_point=anchor_point)
                                info_textbox.append(f"  Applying scale factor {final_scale} for face (1:1.5)...")
                                QApplication.processEvents()
                                crop_img = img[y1:y2, x1:x2]
                                h, w, _ = crop_img.shape
                                if crop_img.size != 0:
                                    cv2.imwrite(f"{edited_folder}/Face-1to1.5/{img_name.split('.')[0]}_{w}x{h}_scale_{final_scale:.2f}.png", crop_img, [cv2.IMWRITE_PNG_COMPRESSION, 4])

                                    # New code to resize and save as 1024x1536
                                    resized_img = cv2.resize(crop_img, (1024, 1536))
                                    cv2.imwrite(f"{edited_folder}/Face-1to1.5/{img_name.split('.')[0]}_1024x1536_scale_{final_scale:.2f}.png", resized_img, [cv2.IMWRITE_PNG_COMPRESSION, 4])

                                    info_textbox.append("  Image saved successfully for face (1:1.5).")
                                    QApplication.processEvents()
                                    info_textbox.append(f"  Final Resolution: {w}x{h}")
                                    QApplication.processEvents()

                    if not face_detected:
                        info_textbox.append("  No face detected.")
                        QApplication.processEvents()

                    info_textbox.append("------")
                    QApplication.processEvents

                    # Convert the cropped OpenCV image to QPixmap and display it (only if crop_img is not None)
                    if crop_img is not None:
                        info_textbox.append(f"crop_img shape: {crop_img.shape}, type: {type(crop_img)}")  # Debug line
                        QApplication.processEvents()  # Make sure to update the UI
                        height, width, channel = crop_img.shape
                        bytesPerLine = 3 * width
                        crop_img = np.ascontiguousarray(crop_img)  # Make sure the array is contiguous in memory
                        qImg = QImage(crop_img.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
                        pixmap = QPixmap.fromImage(qImg)
                        scaled_pixmap = pixmap.scaledToHeight(640, Qt.FastTransformation)  # Scale to a height of 640
                        cropped_image_label.setPixmap(scaled_pixmap)
                    else:
                        info_textbox.append("crop_img is None or empty.")  # Debug line
                        QApplication.processEvents()  # Make sure to update the UI

                    # Convert the original OpenCV image to QPixmap and display it
                    height, width, channel = img.shape
                    bytesPerLine = 3 * width
                    qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
                    pixmap = QPixmap.fromImage(qImg)
                    scaled_pixmap = pixmap.scaledToHeight(640, Qt.FastTransformation)  # Scale to a height of 640
                    original_image_label.setPixmap(scaled_pixmap)
            except Exception as e:
                info_textbox.append(f"An error occurred for image {img_name}: {str(e)}")
                QApplication.processEvents()
                continue  # Skip to the next image
        info_textbox.append("All images have been processed.")
        QApplication.processEvents()

    except Exception as e:
        info_textbox.append(f"An error occurred: {str(e)}")
        QApplication.processEvents()

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'YOLOv8 Cropper'
        self.config_path = 'app_config.txt'
        self.initUI()
        self.load_last_config()

    def initUI(self):
        self.setWindowTitle(self.title)

        # Custom stylesheet for checkboxes
        checkbox_stylesheet = """
        QCheckBox {
            color: white;
            background-color: transparent;
        }
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
            border: 1px solid white;
            background-color: transparent;
        }
        QCheckBox::indicator:checked {
            background-color: white;
        }
        """

        # Create widgets for face model
        self.face_model_button = QPushButton('Select Face YOLO Model', self)
        self.face_model_button.setFixedWidth(150)
        self.face_model_button.clicked.connect(self.load_face_model)
        self.face_model_label = QLabel('Face Model: Not selected', self)

        # Create a horizontal layout for the button and label
        face_model_layout = QHBoxLayout()

        # Add the button and label to the horizontal layout
        face_model_layout.addWidget(self.face_model_button)
        face_model_layout.addWidget(self.face_model_label)

        # Create widgets for body model
        self.body_model_button = QPushButton('Select Body YOLO Model', self)
        self.body_model_button.setFixedWidth(150)
        self.body_model_button.clicked.connect(self.load_body_model)
        self.body_model_label = QLabel('Body Model: Not selected', self)

        # Create a horizontal layout for the button and label
        body_model_layout = QHBoxLayout()

        # Add the button and label to the horizontal layout
        body_model_layout.addWidget(self.body_model_button)
        body_model_layout.addWidget(self.body_model_label)

        # Create widgets for source folder
        self.folder_button = QPushButton('Source Image Folder', self)
        self.folder_button.setFixedWidth(150)
        self.folder_button.clicked.connect(self.load_folder)
        self.folder_label = QLabel('Folder: Not selected', self)

        # Create a horizontal layout for the button and label
        folder_button_layout = QHBoxLayout()

        # Add the button and label to the horizontal layout
        folder_button_layout.addWidget(self.folder_button)
        folder_button_layout.addWidget(self.folder_label)

        # Create run button
        self.run_button = QPushButton('Run Detection and Crop', self)
        self.run_button.setFixedWidth(150)
        self.run_button.clicked.connect(self.run_detection)

        # Create a horizontal layout for centering the run button
        run_button_layout = QHBoxLayout()
        run_button_layout.addStretch(1)
        run_button_layout.addWidget(self.run_button)
        run_button_layout.addStretch(1)

        # Add QTextEdit widget
        self.info_textbox = QTextEdit(self)
        self.info_textbox.setReadOnly(True)

        # Create checkboxes for aspect ratios
        self.aspect_1_1 = QCheckBox('Body 1:1', self)
        self.aspect_1_1.setStyleSheet(checkbox_stylesheet)

        self.aspect_1_1_5 = QCheckBox('Body 1:1.5', self)
        self.aspect_1_1_5.setStyleSheet(checkbox_stylesheet)

        self.face_checkbox = QCheckBox('Face 1:1', self)
        self.face_checkbox.setStyleSheet(checkbox_stylesheet)

        self.face_1_1_5_checkbox = QCheckBox('Face 1:1.5', self)
        self.face_1_1_5_checkbox.setStyleSheet(checkbox_stylesheet)

        self.aspect_boxed = QCheckBox('Body Cropped - Any Ratio', self)
        self.aspect_boxed.setStyleSheet(checkbox_stylesheet)

        self.subject_detection_checkbox = QCheckBox('Subject Detection Results', self)
        self.subject_detection_checkbox.setStyleSheet(checkbox_stylesheet)

        # Create Radio Buttons
        self.radio_group1 = QButtonGroup(self)
        self.radio_group2 = QButtonGroup(self)
        self.radio_group3 = QButtonGroup(self)
        self.radio_group4 = QButtonGroup(self)

        self.head_priority_radio = QRadioButton("Head Priority")
        self.compromise_radio = QRadioButton("Compromise")
        self.radio_group1.addButton(self.head_priority_radio)
        self.radio_group1.addButton(self.compromise_radio)

        self.head_priority_radio2 = QRadioButton("Head Priority")
        self.compromise_radio2 = QRadioButton("Compromise")
        self.radio_group2.addButton(self.head_priority_radio2)
        self.radio_group2.addButton(self.compromise_radio2)

        self.head_priority_radio3 = QRadioButton("Head Priority")
        self.compromise_radio3 = QRadioButton("Face Priority")
        self.radio_group3.addButton(self.head_priority_radio3)
        self.radio_group3.addButton(self.compromise_radio3)

        self.head_priority_radio4 = QRadioButton("Head Priority")
        self.compromise_radio4 = QRadioButton("Face Priority")
        self.radio_group4.addButton(self.head_priority_radio4)
        self.radio_group4.addButton(self.compromise_radio4)

        # Set the default selection
        self.head_priority_radio.setChecked(True)
        self.head_priority_radio2.setChecked(True)
        self.head_priority_radio3.setChecked(True)
        self.head_priority_radio4.setChecked(True)

        # Create a grid layout for checkboxes and radio buttons
        self.aspect_grid = QGridLayout()

        # Add Checkboxes and Radio Buttons to the grid layout
        self.aspect_grid.addWidget(self.aspect_1_1, 0, 0)
        self.aspect_grid.addWidget(self.compromise_radio, 1, 0)
        self.aspect_grid.addWidget(self.head_priority_radio, 2, 0)

        self.aspect_grid.addWidget(self.aspect_1_1_5, 0, 1)
        self.aspect_grid.addWidget(self.compromise_radio2, 1, 1)
        self.aspect_grid.addWidget(self.head_priority_radio2, 2, 1)

        self.aspect_grid.addWidget(self.face_checkbox, 0, 2)
        self.aspect_grid.addWidget(self.compromise_radio3, 1, 2)
        self.aspect_grid.addWidget(self.head_priority_radio3, 2, 2)

        self.aspect_grid.addWidget(self.face_1_1_5_checkbox, 0, 3)
        self.aspect_grid.addWidget(self.compromise_radio4, 1, 3)
        self.aspect_grid.addWidget(self.head_priority_radio4, 2, 3)

        self.aspect_grid.addWidget(self.aspect_boxed, 0, 4)

        self.aspect_grid.addWidget(self.subject_detection_checkbox, 0, 5)

        # Create a progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)  # Set range to 0-100%

        # Create QLabel widgets for the original and cropped images
        self.original_image_label = QLabel('Original Image', self)
        self.cropped_image_label = QLabel('Cropped Image', self)

        # Create Labels to show 'Original' and 'Cropped' text
        original_text_label = QLabel('Original', self)
        cropped_text_label = QLabel('Cropped', self)

        # Create a horizontal spacer
        spacer = QSpacerItem(20, 5, QSizePolicy.Expanding, QSizePolicy.Minimum)

        # Create a new QHBoxLayout for the text labels and add a spacer between them
        text_layout = QHBoxLayout()
        text_layout.addItem(spacer)
        text_layout.addWidget(original_text_label)
        text_layout.addItem(spacer)
        text_layout.addItem(spacer)
        text_layout.addWidget(cropped_text_label)
        text_layout.addItem(spacer)

        # Create QFrames to hold the QLabel widgets for images
        self.original_image_frame = QFrame(self)
        self.original_image_frame.setFrameShape(QFrame.Box)
        self.original_image_frame.setLineWidth(1)
        self.original_image_frame.setFixedSize(650, 650)

        self.cropped_image_frame = QFrame(self)
        self.cropped_image_frame.setFrameShape(QFrame.Box)
        self.cropped_image_frame.setLineWidth(1)
        self.cropped_image_frame.setFixedSize(650, 650)

        # Add QLabel widgets to the frames
        original_image_layout = QVBoxLayout()
        original_image_layout.addWidget(self.original_image_label)
        self.original_image_frame.setLayout(original_image_layout)

        cropped_image_layout = QVBoxLayout()
        cropped_image_layout.addWidget(self.cropped_image_label)
        self.cropped_image_frame.setLayout(cropped_image_layout)

        # Create line separators
        separator1 = QFrame(self)
        separator1.setFrameShape(QFrame.HLine)
        separator1.setFrameShadow(QFrame.Sunken)

        separator2 = QFrame(self)
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)

        separator3 = QFrame(self)
        separator3.setFrameShape(QFrame.HLine)
        separator3.setFrameShadow(QFrame.Sunken)

        separator4 = QFrame(self)
        separator4.setFrameShape(QFrame.HLine)
        separator4.setFrameShadow(QFrame.Sunken)

        separator5 = QFrame(self)
        separator5.setFrameShape(QFrame.HLine)
        separator5.setFrameShadow(QFrame.Sunken)

        # Create label for progress bar
        progress_label = QLabel('Progress:', self)

        # Horizontal layout for the progress bar and its label
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(progress_label)
        progress_layout.addWidget(self.progress_bar)

        # Horizontal layout for the image labels
        image_layout = QHBoxLayout()
        image_layout.addWidget(self.original_image_frame)
        image_layout.addWidget(self.cropped_image_frame)

        # Set layout
        layout = QVBoxLayout()

        # Add widgets and separators to layout

        layout.addLayout(face_model_layout)
        layout.addWidget(separator1)
        layout.addLayout(body_model_layout)
        layout.addWidget(separator2)
        layout.addLayout(folder_button_layout)
        layout.addWidget(separator3)
        layout.addLayout(self.aspect_grid)
        layout.addWidget(separator4)
        layout.addLayout(run_button_layout)  # Add the centered run button layout
        layout.addWidget(separator5)
        layout.addLayout(text_layout)  # Add the text layout for 'Original' and 'Cropped'
        layout.addWidget(separator5)
        layout.addLayout(image_layout)
        layout.addLayout(progress_layout)
        layout.addWidget(self.info_textbox)

        self.setLayout(layout)
        self.show()

    # New method for loading the face model
    def load_face_model(self):
        face_model_path, _ = QFileDialog.getOpenFileName(self, "Load Face Model", "", "All Files (*);;Python Files (*.py)")
        if face_model_path:
            self.FACE_MODEL_PATH = face_model_path
            self.face_model = YOLO(self.FACE_MODEL_PATH)
            self.face_model_label.setText(f'Face Model: {face_model_path}')
            self.save_config()

    # New method for loading the body model
    def load_body_model(self):
        body_model_path, _ = QFileDialog.getOpenFileName(self, "Load Body Model", "", "All Files (*);;Python Files (*.py)")
        if body_model_path:
            self.BODY_MODEL_PATH = body_model_path
            self.body_model = YOLO(self.BODY_MODEL_PATH)
            print(self.body_model.names)  # Print the class names here
            self.body_model_label.setText(f'Body Model: {body_model_path}')
            self.save_config()

    def load_last_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    self.BODY_MODEL_PATH = lines[0].strip()
                    self.FACE_MODEL_PATH = lines[1].strip()
                    self.body_model = YOLO(self.BODY_MODEL_PATH)
                    self.face_model = YOLO(self.FACE_MODEL_PATH)
                    self.body_model_label.setText(f'Body Model: {self.BODY_MODEL_PATH}')
                    self.face_model_label.setText(f'Face Model: {self.FACE_MODEL_PATH}')
                if len(lines) > 2:
                    self.SOURCE_FOLDER = lines[2].strip()
                    self.folder_label.setText(f'Folder: {self.SOURCE_FOLDER}')  # Update folder label

    def save_config(self):
        with open(self.config_path, 'w') as f:
            if hasattr(self, 'BODY_MODEL_PATH'):
                f.write(f"{self.BODY_MODEL_PATH}\n")
            else:
                f.write("\n")

            if hasattr(self, 'FACE_MODEL_PATH'):
                f.write(f"{self.FACE_MODEL_PATH}\n")
            else:
                f.write("\n")

            if hasattr(self, 'SOURCE_FOLDER'):
                f.write(f"{self.SOURCE_FOLDER}\n")  # Save source folder
            else:
                f.write("\n")

    def load_model(self):
        body_model_path, _ = QFileDialog.getOpenFileName(self, "Load Body Model", "", "All Files (*);;Python Files (*.py)")
        face_model_path, _ = QFileDialog.getOpenFileName(self, "Load Face Model", "", "All Files (*);;Python Files (*.py)")

        if body_model_path and face_model_path:
            self.BODY_MODEL_PATH = body_model_path
            self.FACE_MODEL_PATH = face_model_path
            self.body_model = YOLO(self.BODY_MODEL_PATH)
            self.face_model = YOLO(self.FACE_MODEL_PATH)
            self.save_config()

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.SOURCE_FOLDER = folder
            self.folder_label.setText(f'Folder: {folder}')
            self.save_config()

    def run_detection(self):
        if hasattr(self, 'BODY_MODEL_PATH') and hasattr(self, 'FACE_MODEL_PATH') and hasattr(self, 'SOURCE_FOLDER'):
            aspect_ratios = self.get_aspect_ratios()

            # Print out the aspect_ratios to see what's checked
            print("Aspect Ratios Checked:", aspect_ratios)

            detect_and_crop(self.body_model, self.face_model, self.SOURCE_FOLDER, self.info_textbox, aspect_ratios,self.progress_bar,self.original_image_label, self.cropped_image_label, self.head_priority_radio, self.head_priority_radio2,self.head_priority_radio3,self.head_priority_radio4)

        else:
            print("Please select both models and image folder first.")

    def get_aspect_ratios(self):
        aspect_ratios = {
            'body_1_1': (self.aspect_1_1.isChecked(),),
            'body_1_1_5': (self.aspect_1_1_5.isChecked(),),
            'face_1_1': (self.face_checkbox.isChecked(),),
            'face_1_1_5': (self.face_1_1_5_checkbox.isChecked(),),
            'body_cropped_any': (self.aspect_boxed.isChecked(),),
            'subject_detection_results': (self.subject_detection_checkbox.isChecked(),)
        }
        return aspect_ratios

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
