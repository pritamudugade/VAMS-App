from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile
from pathlib import Path
import config

def _display_detected_frames(conf, model, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: None
    """
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # Predict the objects in the image using YOLOv8 model
    res = model.predict(image, conf=conf)

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    
    # Update the displayed frame with detected objects
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True)

def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
      model_path (str): The path to the YOLO model file.

    Returns:
      A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model

def infer_uploaded_image(conf, model):
    """
    Execute inference for uploaded image
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            # adding the uploaded image to the page with caption
            st.image(
                image=source_img,
                caption="Uploaded Image",
                use_column_width=True
            )

    if source_img:
        if st.button("Execution"):
            with st.spinner("Running..."):
                res = model.predict(uploaded_image, conf=conf)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]

            with col2:
                st.image(res_plotted,
                          caption="Detected Image",
                          use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.xywh)
                except Exception as ex:
                    st.write("No image is uploaded yet!")
                    st.write(ex)

def infer_uploaded_video(confidence, model):
    """
    Execute inference for uploaded video
    :param confidence: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_video = st.sidebar.file_uploader(
        label="Choose a video..."
    )

    if source_video:
        st.video(source_video)

        # Convert uploaded video file to string path
        uploaded_video_path = tempfile.NamedTemporaryFile(delete=False)
        uploaded_video_path.write(source_video.read())
        uploaded_video_path = uploaded_video_path.name

        # Open the uploaded video file
        vid_cap = cv2.VideoCapture(uploaded_video_path)

        # Get the frame rate of the input video
        frame_rate = int(vid_cap.get(cv2.CAP_PROP_FPS))

        # Create a list to store frames with detected objects
        frames_with_objects = []

        while vid_cap.isOpened():
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(confidence, model, st.empty(), image)

                # Store the frame with detected objects in the list
                frames_with_objects.append(image)
                
                # Add a delay to match the frame rate of the input video
                cv2.waitKey(int(1000 / frame_rate))
            else:
                vid_cap.release()
                break

        # Release video objects
        cv2.destroyAllWindows()

        # Create an output video writer with the same frame rate
        output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (720, int(720 * (9 / 16))))

        # Write frames with detected objects to the output video
        for frame in frames_with_objects:
            out.write(frame)

        # Release the output video writer
        out.release()

        # Display a link to download the processed video
        st.markdown(f"Download the processed video [here]({output_video_path})")

        # Add a button to save the processed video
        if st.button("Save Processed Video"):
            with open(output_video_path, "rb") as f:
                st.download_button(
                    label="Click here to download the processed video",
                    data=f,
                    key="processed_video",
                    file_name="processed_video.avi",
                )



def infer_uploaded_webcam(conf, model):
    """
    Execute inference for uploaded webcam
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if success:
            _display_detected_frames(conf, model, st.empty(), st.empty(), image)
        else:
            cap.release()
            break
