import streamlit as st
from video_processing import process_video
import tempfile

def main():
  st.title('Human Activity Recognition')

  # File uploader for input video
  uploaded_file = st.file_uploader("Upload Video", type=['mp4'])

  if uploaded_file is not None:
    # Display uploaded video
    st.video(uploaded_file)

    # Process video on button click
    if st.button('Process Video'):
      with st.spinner('Processing...'):
        temp_video_file = tempfile.NamedTemporaryFile(delete=False)
        temp_video_file.write(uploaded_file.read())

        output_video_path = process_video(temp_video_file.name)
        st.video(output_video_path)  # Display the output video with class names

if __name__ == "__main__":
  main()
