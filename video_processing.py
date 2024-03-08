import cv2
import os
from collections import deque
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TimeDistributed

# Load the pre-trained model with custom objects (replace paths accordingly)
custom_optimizer = Adam()
convlstm = load_model('F:/Major Project/Streamlit UI/model/convlstm_model.h5', custom_objects={'TimeDistributed': TimeDistributed, 'Adam': custom_optimizer})

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20
CLASSES_LIST = ["JumpRope", "HorseRace", "JavelinThrow", "TennisSwing"]

def process_video(input_video_path):
  # Initialize the VideoCapture object to read from the video file.
  video_reader = cv2.VideoCapture(input_video_path)

  # Get the width and height of the video.
  original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
  original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

  # Initialize the VideoWriter Object to store the output video in the disk.
  output_video_path = input_video_path.replace('.', '_output.')
  video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                 video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

  # Declare a queue to store video frames.
  frames_queue = deque(maxlen=SEQUENCE_LENGTH)

  # Iterate until the video is accessed successfully.
  while video_reader.isOpened():

    # Read the frame.
    ok, frame = video_reader.read()

    # Check if frame is not read properly then break the loop.
    if not ok:
      break

    # Resize the Frame to fixed Dimensions.
    resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

    # Normalize the resized frame by dividing it with 255.
    normalized_frame = resized_frame / 255

    # Append the pre-processed frame into the frames list.
    frames_queue.append(normalized_frame)

    # Check if the number of frames in the queue are equal to the fixed sequence length.
    if len(frames_queue) == SEQUENCE_LENGTH:
      # Pass the normalized frames to the model and get the predicted probabilities.
      predicted_labels_probabilities = convlstm.predict(np.expand_dims(frames_queue, axis=0))[0]

      # Get the index of class with highest probability.
      predicted_label = np.argmax(predicted_labels_probabilities)

      # Get the class name using the retrieved index.
      predicted_class_name = CLASSES_LIST[predicted_label]

      # Write predicted class name on top of a copy of the frame
      frame_with_text = frame.copy()
      cv2.putText(frame_with_text, predicted_class_name, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 4)

      # Write the frame with text into the disk
      video_writer.write(frame_with_text)

  # Release the VideoCapture and VideoWriter objects.
  video_reader.release()
  video_writer.release()

  return output_video_path
