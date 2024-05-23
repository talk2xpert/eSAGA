import cv2
import  os
import datetime

class image_capturing:


    def start_capturing_images_from_vcam(image_path,f_count=5):
        time = 1000
        # def start_capturing_images_from_vcam(image_path,time):
        try:
            print("Starting The Video")
            video_capture = cv2.VideoCapture(0)
            frame_count = 0  # Counter to track the frames
            print("Video Capture", frame_count)
            while frame_count <= f_count:
                ret, frame = video_capture.read()
                print(ret)
                print(frame)
                if not ret:
                    break

                # Save each frame to a file
                frame_count += 1
                # Generate a unique filename using timestamp and UUID
                timestamp = datetime.datetime
                # unique_id = str(uuid.uuid4())[:8]
                # Using the first 8 characters of a UUID
                # filename = f'frame_{timestamp}_{unique_id}.jpg'
                filename = f'{frame_count}.jpg'
                frame_name = os.path.join(image_path, filename)
                # frame_name = f'{save_path}/frame_{frame_count}.jpg'
                cv2.imwrite(frame_name, frame)

                # Display the captured frame (optional)
                cv2.imshow('Frame', frame)
                if cv2.waitKey(time) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print("Error:", e)
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
