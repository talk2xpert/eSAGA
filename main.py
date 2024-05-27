import os.path

from org.saga.byop.production.utility.image_capturing import image_capturing
from org.saga.byop.production.utility.image_validator import  image_validator
from org.saga.byop.production.utility.helper import  helper
from org.saga.byop.training import face_detection
import logging
import config_manager

class main:
    # Set up logging
    def __init__(self):
        # log_setup.log_setup()
        self.logger = logging.getLogger(__name__)

    def run(self,candidate_name):

        voting_repository_path = config_manager.get_voting_repository_path()
        candidate_image_directory_path = os.path.join(voting_repository_path, candidate_name)

        # 1- getting 10 images from frames and saving in candidate_repository
        #image_capturing.start_capturing_images_from_vcam(candidate_directory_path,10)

        fd = face_detection.face_detection()
        blink_results=fd.face_detector(config_manager.get_frame_count(),candidate_image_directory_path,True,True)
        most_common_value, count = helper.likelihood_estimator(blink_results)
        print(f"** The value that occurs the most is: {most_common_value} with {count} occurrences. **")

        # reading the captured images of the candidate
        image_paths = helper.load_images_from_dir(candidate_image_directory_path)
        # print(image_paths)


        # 2- DO SPOOF CHECK
        most_common_value,count = image_validator.image_spoof_check(image_paths)
        print(f"SPOOF DETECTION STAGE [RESULT] : The most predicted outcome - {most_common_value} with {count} occurrences.")

        if(most_common_value=="SPOOF"):
            print("SPOOF DETECTION STAGE [RESULT] : FAILED FACELiveliness Detection test")
            # print("**********************PLEASE EXIT*************************")
        else:
            print("SPOOF DETECTION STAGE [RESULT] : PASSED FACELiveliness Detection test")


        # 3-DO SIMILARITY CHECK
        reference_image_repository_path = config_manager.get_candidate_repository_path()
        reference_image_path=os.path.join(reference_image_repository_path,candidate_name + ".jpg")
        # print(reference_image_path)
        most_common_value,count=image_validator.image_similarity_check(image_paths,reference_image_path)
        print(f"SIMILARITY CHECK [RESULT] : The value that occurs the most is: {most_common_value} with {count} occurrences.")

        bool1 = True
        bool2 = False

        if (most_common_value==bool1):
            print("PASSED : Congratulations  ......... PASSED SIMILARITY TEST ")
            # print("**********************PLEASE CONTINUE*************************")
        elif( most_common_value==bool2):
            print("FAILED: OOPS .................. FAILED SIMILARITY TEST ")
            #print("**********************PLEASE EXIT*************************")

        else:
            print("ERROR : Retry  ..... FAILED SIMILARITY TEST AS FACE NOT DETECTED ")
            # print("**********************PLEASE EXIT*************************")




m = main()
m.run("sanjay")