import cv2
import imageio.v2 as imageio
import imageio as iio_main

def generate_video():
    # Read the first frame
    first_frame = imageio.imread('/app/first_frame.png')
    
    # Initialize the video writer with required properties
    writer = iio_main.get_writer(
        '/app/output/video.mp4', 
        fps=16, 
        codec='libx264', 
        pixelformat='yuv420p',
        macro_block_size=None
    )
    
    # Circle parameters for the Chinese character "美"
    # Found dynamically via bounding box checking
    center = (588, 704)
    radius = 95
    thickness = 6
    num_frames = 48
    
    for i in range(num_frames):
        img = first_frame.copy()
        
        # Calculate how much of the circle to draw
        # Frame 0 is untouched, Frame 47 has the full circle
        progress = i / (num_frames - 1)
        end_angle_diff = 360 * progress
        
        # Draw the arc from the top (-90 degrees) clockwise
        if end_angle_diff > 0:
            end_angle = -90 + end_angle_diff
            cv2.ellipse(
                img, 
                center, 
                (radius, radius), 
                0, 
                -90, 
                end_angle, 
                (255, 0, 0), # Red color in RGB format
                thickness, 
                cv2.LINE_AA
            )
            
        writer.append_data(img)
        
    writer.close()

if __name__ == '__main__':
    generate_video()
