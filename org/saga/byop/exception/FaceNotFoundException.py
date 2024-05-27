class FaceNotFoundException(Exception):
    """Custom exception raised when no face is found in the frame."""
    def __init__(self, message="No face found in the frame"):
        self.message = message
        super().__init__(self.message)