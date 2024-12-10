import cv2
import numpy as np
from src.hand_detection import HandDetector
from src.canvas import Canvas
from models.google_api import GeminiAPI
from config.settings import API_KEY

class Canvas:
    def __init__(self):
        self.canvas_image = None  # Initialize with None

    def initialize_canvas(self, frame):
        # Create a canvas with the same size as the frame
        self.canvas_image = np.zeros_like(frame)

    def draw(self, position):
        # Draw a dot at the current position
        cv2.circle(self.canvas_image, position, 5, (255, 0, 0), -1)  # Drawing circles as dots

    def draw_line(self, start_position, end_position):
        # Draw a line between the previous and current finger positions
        cv2.line(self.canvas_image, start_position, end_position, (255, 0, 0), 5)  # Draw a smooth line

    def reset(self):
        # Clear the canvas by setting it to a blank state
        self.canvas_image = np.zeros_like(self.canvas_image)

    def get_combined_frame(self, frame):
        # Ensure the canvas has the same size as the frame
        if self.canvas_image is None:
            self.initialize_canvas(frame)
        # Combine the canvas image with the webcam feed
        return cv2.addWeighted(frame, 0.5, self.canvas_image, 0.5, 0)  # Blended frame and canvas

    def get_image(self):
        return self.canvas_image


def main():
    hand_detector = HandDetector()  # Initialize hand detector
    gemini_api = GeminiAPI(api_key="AIzaSyCwn7pyx7X0VMkU3OzdtK9vGoJoBE_nSwU")
    canvas = Canvas()
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Virtual Calculator', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Virtual Calculator', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    previous_position = None  # Store previous fingertip position for continuous drawing
    equation_finished = False  # Flag to check if the equation drawing is finished
    processing = False  # New flag to track API call status
    result = ""  # Variable to store the equation result

    # Define the clickable box area for starting processing
    box_top_left = (50, 50)  # Top-left corner of the clickable box
    box_bottom_right = (150, 150)  # Bottom-right corner of the clickable box

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame to fix the mirrored view
        frame = cv2.flip(frame, 1)

        # Draw the clickable box on the frame
        cv2.rectangle(frame, box_top_left, box_bottom_right, (0, 255, 0), 2)
        cv2.putText(frame, "Start", (box_top_left[0] + 10, box_top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Detect hand gestures and get fingertip position
        frame = hand_detector.detect(frame)
        gestures = hand_detector.get_gestures()

        # Using get_finger_tip_position to check index finger position only
        index_finger = hand_detector.get_finger_tip_position()

        # Perform actions based on gestures
        if gestures == 'index_finger_up' and not equation_finished:
            if index_finger:
                # Draw a continuous line from previous position to current position
                if previous_position:
                    canvas.draw_line(previous_position, index_finger)  # Draw a smooth line
                previous_position = index_finger  # Update the previous position
        elif gestures == 'thumb_up':
            # Clear the canvas when thumb is up
            canvas.reset()
            previous_position = None
            result = ""  # Clear the result as well
            processing = False
            equation_finished = False
        else:
            previous_position = None  # If no gesture or different gesture, reset previous position

        # Check if index finger is in the clickable box area to start processing
        if index_finger:
            if (box_top_left[0] <= index_finger[0] <= box_bottom_right[0] and
                box_top_left[1] <= index_finger[1] <= box_bottom_right[1] and
                not processing):

                processing = True  # Set processing flag to True
                equation_image = canvas.get_image()  # Capture the canvas image for API processing

                try:
                    result = gemini_api.solve(equation_image)  # Get the equation result from the API
                except Exception as e:
                    result = f"Error: {str(e)}"
                
                processing = False
                equation_finished = True  # Mark the equation as finished

        combined_frame = canvas.get_combined_frame(frame)  # Pass frame here

        # Create a white space for displaying the answer on the right side
        answer_area = np.zeros((combined_frame.shape[0], 300, 3), dtype=np.uint8) + 255
        combined_frame = np.hstack((combined_frame, answer_area))

        # Show "Processing..." or the result in the answer area
        if processing:
            cv2.putText(combined_frame, "Processing...", (combined_frame.shape[1] - 290, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        elif equation_finished:
            y0, dy = 30, 20  # Adjust dy for smaller font size
            wrapped_result = wrap_text(result, 25)  # Adjust max_width for smaller font size
            for i, line in enumerate(wrapped_result):
                y = y0 + i * dy
                cv2.putText(combined_frame, line, (combined_frame.shape[1] - 290, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imshow('Virtual Calculator', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def wrap_text(text, max_width):
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) + 1 <= max_width:
            if current_line:
                current_line += " " + word
            else:
                current_line = word
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines

if __name__ == '__main__':
    main()
